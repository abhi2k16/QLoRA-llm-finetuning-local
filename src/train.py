"""
train.py — QLoRA fine-tuning pipeline. 
Uses:  transformers + peft + bitsandbytes + torch (no Unsloth, no trl)
Stack: HuggingFace transformers + PEFT + bitsandbytes (no Unsloth).

Run from the project root:
    python src/train.py
    or
    python -m src.train

Requirements:
1. Install the required packages:
    pip install torch transformers peft bitsandbytes datasets pyyaml
2. Ensure you have a compatible NVIDIA GPU with CUDA support and the appropriate drivers installed.
3. Prepare your dataset in the expected format (e.g., a JSON file with conversations).
4. Configure the training parameters in the config/qlora_config.yaml file, including model name, LoRA settings, and training hyperparameters.
Ensure the win_llm conda environment is active before running.
"""

import logging
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from dataset import ShareGPTDataset

# ── Logging ───────────────────────────────────────────────────────────────────
Path("outputs").mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("outputs/train.log", mode="w"),
    ],
)
logger = logging.getLogger(__name__)

CONFIG_PATH = Path("config/qlora_config.yaml")
DATA_PATH   = Path("data/raw/dataset.json")
SEED        = 42


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config not found at '{CONFIG_PATH}'.")
    with open(CONFIG_PATH, "r") as f: # Use safe_load to avoid potential security issues with untrusted YAML files
        cfg = yaml.safe_load(f)       # Load the YAML file into a Python dictionary 
    logger.info("Config loaded from %s", CONFIG_PATH) # Log the successful loading of the configuration file, including the path from which it was loaded
    return cfg


def check_gpu() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. "
            "Re-install PyTorch with: "
            "pip install torch --index-url https://download.pytorch.org/whl/cu121"
        )
    name  = torch.cuda.get_device_name(0)
    vram  = torch.cuda.get_device_properties(0).total_memory / 1e9
    logger.info("GPU: %s | VRAM: %.2f GB", name, vram)
    if vram < 3.0:
        logger.warning("Less than 3 GB VRAM detected (%.2f GB). OOM is possible.", vram)


def load_tokenizer(model_name: str):
    """
    Load the tokenizer and ensure it has a pad token. If not, set the pad token to be the same as 
    the tokenizer'sthe end-of-sequence token. This is important for proper padding during training, 
    and particularly when dealing with variable-length sequences in a batch. This ensures that the 
    tokenizer can handle variable-length sequences especially when using causal language models that 
    may not have a dedicated pad token.
    function inputs are:
    - model_name: The name or path of the pre-trained model to load (e.g., "gpt2", "EleutherAI/gpt-j-6B").
    returns:
    - The loaded tokenizer, with a pad token guaranteed to be set. 
    """
    from transformers import AutoTokenizer
    logger.info("Loading tokenizer: %s", model_name)
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token 
        logger.info("pad_token set to eos_token")
    return tok


def load_model_4bit(model_name: str):
    """
    Load the model in 4-bit precision using bitsandbytes. This significantly reduces VRAM usage, 
    allowing us to fine-tune larger models on limited hardware. The BitsAndBytesConfig is set to 
    use NormalFloat4 quantization, which provides better quality than other 4-bit quantization types. 
    The compute dtype is set to float16, which is compatible with most GPUs and helps maintain 
    performance while reducing memory usage. Additionally, double quantization is enabled for extra 
    memory savings. The model is loaded with device_map="auto" to automatically place layers on the 
    appropriate devices (e.g., GPU), and use_cache is set to False to allow for gradient checkpointing 
    during training.
    function inputs are:
    - model_name: The name or path of the pre-trained model to load (e.g., "gpt2", "EleutherAI/gpt-j-6B"). 
    returns:
    - model: The loaded model in 4-bit precision, ready for fine-tuning with LoRA adapters.
    """
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    logger.info("Loading model in 4-bit: %s", model_name)
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit              = True,
        bnb_4bit_quant_type       = "nf4",          # NormalFloat4 — best quality
        bnb_4bit_compute_dtype    = torch.float16,  # FP16 compute (GTX 1050 safe)
        bnb_4bit_use_double_quant = True,           # Extra 0.4 bpw saving
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config = bnb_cfg,
        device_map          = "auto",
    )
    model.config.use_cache = False       # Must be False for gradient checkpointing
    model.enable_input_require_grads()   # Required for PEFT to attach adapters
    logger.info("Model loaded successfully")
    return model


def attach_lora(model, lora_cfg: dict):
    """
    Attach LoRA adapters to the model. This allows for efficient fine-tuning by only training
    a small number of additional parameters, while keeping the base model frozen. The LoRA
    configuration is defined by the lora_cfg dictionary, which specifies the rank (r), scaling 
    factor (alpha), dropout, and target modules for the adapters. After attaching the adapters, 
    which modules to attach to, and what hyperparameters to use for the adapters. The function then
    uses get_peft_model to apply the LoRA configuration to the model. Finally, the function calculates
    and logs the number of trainable parameters in the model. This is useful for understanding how
    many parameters are being trained during the fine-tuning process, and whether the LoRA adapters
    are configured correctly. The function returns the model with the LoRA adapters attached. After
    attaching the adapters, the function calculates the number of trainable parameters in the model
    and logs this information. This is useful for understanding the efficiency of the fine-tuning
    process, as we only want to train a small number of additional parameters (the LoRA adapters)
    rather than the entire model. The function also logs the total number of parameters in the model
    for comparison. 
    The function inputs are: 
    - model: The base model to which to attach LoRA adapters
    - lora_cfg: A dictionary containing the LoRA configuration parameters`
    returns:
    - The model with LoRA adapters attached, ready for fine-tuning.
    """
    from peft import LoraConfig, get_peft_model, TaskType
    logger.info(
        "Attaching LoRA: r=%d, alpha=%d, dropout=%s, modules=%s",
        lora_cfg["r"], lora_cfg["alpha"], lora_cfg["dropout"],
        lora_cfg["target_modules"],
    )
    config = LoraConfig(
        r              = lora_cfg["r"],
        lora_alpha     = lora_cfg["alpha"],
        target_modules = lora_cfg["target_modules"],
        lora_dropout   = lora_cfg["dropout"],
        bias           = "none",
        task_type      = TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    logger.info(
        "Trainable parameters: %s / %s (%.3f%%)",
        f"{trainable:,}", f"{total:,}", 100 * trainable / total,
    )
    return model


def collate_fn(batch, pad_token_id: int):
    """Right-pad a batch of token-ID tensors to equal length."""
    max_len = max(x.size(0) for x in batch)
    padded  = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    for i, seq in enumerate(batch):
        padded[i, : seq.size(0)] = seq
    return padded


def log_vram(label: str) -> None:
    """Log current VRAM usage. Useful for debugging OOM issues."""
    if not torch.cuda.is_available():
        return
    alloc = torch.cuda.memory_allocated() / 1e9
    resv  = torch.cuda.memory_reserved()  / 1e9
    logger.info("VRAM [%s] — allocated: %.2f GB  reserved: %.2f GB", label, alloc, resv)


# ── Training loop ─────────────────────────────────────────────────────────────

class InMemoryTextDataset(torch.utils.data.Dataset):
    """Wraps pre-tokenised examples for the DataLoader to fetch. This is more efficient than tokenising on-the-fly.
    The dataset takes a list of raw text strings and a tokenizer, and pre-tokenizes all the examples in memory during 
    initialization. Each text string is tokenized using the provided tokenizer, with truncation and padding to a 
    specified maximum length. The resulting token ID tensors are stored in a list for efficient retrieval during 
    training. The __len__ method returns the number of examples in the dataset, while the __getitem__ method 
    retrieves the pre-tokenized tensor for a given index. This approach allows for faster data loading during training, 
    as the tokenization step is performed once upfront rather than on-the-fly for each batch."""
    def __init__(self, tokenizer, texts: list[str], max_length: int):
        self.examples = []
        for text in texts:
            enc = tokenizer(
                text,
                truncation    = True,
                max_length    = max_length,
                return_tensors= "pt",
            )
            self.examples.append(enc["input_ids"].squeeze(0))

    def __len__(self):         return len(self.examples)
    def __getitem__(self, i):  return self.examples[i]


def run_training(model, tokenizer, texts: list[str], cfg: dict, device: torch.device) -> float:
    """Run the training loop. Returns the final average loss.
    function inputs are:
    - model: The model to be fine-tuned, with LoRA adapters attached.
    - tokenizer: The tokenizer corresponding to the base model, used for tokenizing the training texts.
    - texts: A list of raw text strings that will be used for training. These will be tokenized and fed into the model during training.
    - cfg: A configuration dictionary containing training hyperparameters such as max_steps, learning_rate, and gradient_accumulation_steps.
    - device: The torch.device on which to perform training (e.g., "cuda" for GPU). 
    returns:
    - The final average loss after training is complete. 
    """
    import bitsandbytes as bnb

    tcfg       = cfg["training"]
    max_len    = cfg["model"]["max_seq_length"]
    max_steps  = tcfg["max_steps"]
    grad_accum = tcfg["gradient_accumulation_steps"]
    lr         = float(tcfg["learning_rate"])

    # Dataset & loader
    torch_ds = InMemoryTextDataset(tokenizer, texts, max_len)
    loader   = DataLoader(
        torch_ds,
        batch_size  = 1,
        shuffle     = True,
        collate_fn  = lambda b: collate_fn(b, tokenizer.pad_token_id),
    )

    # Optimizer — paged 8-bit AdamW offloads state to CPU RAM
    optimizer = bnb.optim.PagedAdamW8bit(
        [p for p in model.parameters() if p.requires_grad],
        lr = lr,
    )

    # LR warmup scheduler
    from torch.optim.lr_scheduler import LinearLR
    warmup_steps = tcfg.get("warmup_steps", 5)
    scheduler = LinearLR(
        optimizer,
        start_factor = 0.1,
        end_factor   = 1.0,
        total_iters  = warmup_steps,
    )

    # FP16 gradient scaler
    scaler = torch.cuda.amp.GradScaler()

    model.train()
    optimizer.zero_grad()

    step       = 0
    accum_loss = 0.0
    last_loss  = 0.0

    logger.info("Training started — %d steps, grad_accum=%d", max_steps, grad_accum)

    while step < max_steps:
        for batch in loader:
            if step >= max_steps:
                break

            input_ids = batch.to(device)
            labels    = input_ids.clone()
            # Mask padding tokens so they don't contribute to the loss
            labels[labels == tokenizer.pad_token_id] = -100

            with torch.cuda.amp.autocast(dtype=torch.float16):
                outputs = model(input_ids=input_ids, labels=labels)
                loss    = outputs.loss / grad_accum

            scaler.scale(loss).backward()
            accum_loss += loss.item()

            if (step + 1) % grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if step < warmup_steps:
                    scheduler.step()

                last_loss = accum_loss
                logger.info(
                    "step %d/%d | loss: %.4f | lr: %.2e",
                    step + 1, max_steps, accum_loss,
                    optimizer.param_groups[0]["lr"],
                )
                accum_loss = 0.0

            step += 1

    return last_loss


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("=" * 60)
    logger.info("  QLoRA Fine-Tuning  (HuggingFace + PEFT + bitsandbytes)")
    logger.info("=" * 60)

    check_gpu()
    device = torch.device("cuda")
    cfg    = load_config()

    tokenizer = load_tokenizer(cfg["model"]["base_model_name"])
    model     = load_model_4bit(cfg["model"]["base_model_name"])
    model     = attach_lora(model, cfg["lora"])

    # Prepare dataset
    logger.info("Loading dataset from %s", DATA_PATH)
    ds_wrapper = ShareGPTDataset(
        tokenizer      = tokenizer,
        max_seq_length = cfg["model"]["max_seq_length"],
    )

    eval_cfg   = cfg.get("evaluation", {})
    eval_ratio = float(eval_cfg.get("eval_ratio", 0.0))
    splits     = ds_wrapper.prepare_splits(str(DATA_PATH), eval_ratio=eval_ratio, seed=SEED)
    train_texts = splits["train"]["text"]

    log_vram("before training")

    # Train
    final_loss = run_training(model, tokenizer, train_texts, cfg, device)
    logger.info("Training complete. Final loss: %.4f", final_loss)

    # Optional evaluation
    if "eval" in splits:
        logger.info("Evaluating on held-out split (%d examples)...", splits["eval"].num_rows)
        model.eval()
        eval_losses = []
        eval_texts  = splits["eval"]["text"]
        eval_ds     = InMemoryTextDataset(tokenizer, eval_texts, cfg["model"]["max_seq_length"])
        eval_loader = DataLoader(
            eval_ds,
            batch_size = 1,
            collate_fn = lambda b: collate_fn(b, tokenizer.pad_token_id),
        )
        with torch.no_grad():
            for batch in eval_loader:
                input_ids = batch.to(device)
                labels    = input_ids.clone()
                labels[labels == tokenizer.pad_token_id] = -100
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    out = model(input_ids=input_ids, labels=labels)
                eval_losses.append(out.loss.item())
        avg_eval = sum(eval_losses) / len(eval_losses)
        logger.info("Eval loss: %.4f", avg_eval)

    # Peak VRAM
    peak = torch.cuda.max_memory_allocated() / 1e9
    logger.info("Peak VRAM: %.2f GB", peak)

    # Save adapter
    save_path = Path(cfg["training"]["output_dir"]) / "final_model"
    save_path.mkdir(parents=True, exist_ok=True)
    logger.info("Saving adapter to %s ...", save_path)
    model.save_pretrained(str(save_path))
    tokenizer.save_pretrained(str(save_path))

    logger.info("=" * 60)
    logger.info("  Done! Adapter saved to: %s", save_path)
    logger.info("  Inference: python src/inference.py --prompt \"Your prompt\"")
    logger.info("  Chat:      python src/chat.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
