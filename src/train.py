"""
train.py — QLoRA fine-tuning pipeline.

Stack: HuggingFace transformers + PEFT + bitsandbytes (no Unsloth).

Run from the project root:
    python src/train.py

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
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    logger.info("Config loaded from %s", CONFIG_PATH)
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
    from transformers import AutoTokenizer
    logger.info("Loading tokenizer: %s", model_name)
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        logger.info("pad_token set to eos_token")
    return tok


def load_model_4bit(model_name: str):
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    logger.info("Loading model in 4-bit: %s", model_name)
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit              = True,
        bnb_4bit_quant_type       = "nf4",         # NormalFloat4 — best quality
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
    if not torch.cuda.is_available():
        return
    alloc = torch.cuda.memory_allocated() / 1e9
    resv  = torch.cuda.memory_reserved()  / 1e9
    logger.info("VRAM [%s] — allocated: %.2f GB  reserved: %.2f GB", label, alloc, resv)


# ── Training loop ─────────────────────────────────────────────────────────────

class InMemoryTextDataset(torch.utils.data.Dataset):
    """Wraps pre-tokenised examples for the DataLoader."""
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
    """Run the training loop. Returns the final average loss."""
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
