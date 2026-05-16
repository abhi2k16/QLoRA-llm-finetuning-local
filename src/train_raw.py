"""
train_raw.py — Raw PyTorch training loop, no trl / SFTTrainer dependency.

If you want the simplest non-Unsloth training path, use this script.

Run from project root:
    python src/train_raw.py
"""

import json
import logging
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Dataset as TorchDataset

sys.path.insert(0, str(Path(__file__).parent))

from runtime_checks import require_cuda_runtime

# ── Logging ──────────────────────────────────────────────────────────────────
Path("outputs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("outputs/train_raw.log", mode="w"),
    ],
)
logger = logging.getLogger(__name__)


# ── Config ───────────────────────────────────────────────────────────────────
CONFIG_PATH = Path("config/qlora_config.yaml")
DATA_PATH   = Path("data/raw/dataset.json")


def load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


# ── Dataset ──────────────────────────────────────────────────────────────────
class ShareGPTDataset(TorchDataset):
    """Tokenises a ShareGPT JSON file into input_ids tensors."""

    def __init__(self, json_path: Path, tokenizer, max_length: int):
        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        self.examples = []
        for item in raw:
            messages = []
            for turn in item.get("conversations", []):
                role  = "user" if turn["from"] == "human" else "assistant"
                messages.append({"role": role, "content": turn["value"]})

            if not messages:
                continue

            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            tokens = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            input_ids = tokens["input_ids"].squeeze(0)
            self.examples.append(input_ids)

        logger.info(f"Loaded {len(self.examples)} training examples")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate_fn(batch, pad_token_id: int):
    """Left-pad sequences in a batch to the same length."""
    max_len = max(x.size(0) for x in batch)
    padded  = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    for i, seq in enumerate(batch):
        padded[i, -seq.size(0):] = seq
    return padded


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    logger.info("=" * 60)
    logger.info("  Raw PyTorch QLoRA Training (HuggingFace + PEFT)")
    logger.info("=" * 60)

    cfg = load_config()
    model_cfg    = cfg["model"]
    lora_cfg     = cfg["lora"]
    train_cfg    = cfg["training"]

    max_seq_length = model_cfg["max_seq_length"]
    max_steps      = train_cfg["max_steps"]
    lr             = float(train_cfg["learning_rate"])
    grad_accum     = train_cfg["gradient_accumulation_steps"]

    # ── Step 1: GPU check ────────────────────────────────────────────────────
    require_cuda_runtime("train_raw.py")
    device = torch.device("cuda")
    logger.info(f"GPU: {torch.cuda.get_device_name(0)} | "
                f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # ── Step 2: Load model ───────────────────────────────────────────────────
    logger.info("Loading tokenizer and model with 4-bit quantization...")
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, TaskType, get_peft_model

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["base_model_name"])
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["base_model_name"],
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.config.use_cache = False
    model.enable_input_require_grads()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info("Model loaded OK")

    # ── Step 3: Attach LoRA ──────────────────────────────────────────────────
    logger.info("Attaching LoRA adapters...")
    lora_config = LoraConfig(
        r=lora_cfg["r"],
        target_modules=lora_cfg["target_modules"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(
        model,
        lora_config,
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {trainable:,}")

    # ── Step 4: Dataset ──────────────────────────────────────────────────────
    logger.info("Preparing dataset...")
    dataset = ShareGPTDataset(DATA_PATH, tokenizer, max_seq_length)
    pad_id  = tokenizer.pad_token_id

    loader = DataLoader(
        dataset,
        batch_size  = 1,
        shuffle     = True,
        collate_fn  = lambda b: collate_fn(b, pad_id),
    )

    # ── Step 5: Optimizer ────────────────────────────────────────────────────
    logger.info("Setting up 8-bit paged AdamW optimizer...")
    import bitsandbytes as bnb
    optimizer = bnb.optim.PagedAdamW8bit(
        [p for p in model.parameters() if p.requires_grad],
        lr = lr,
    )

    # ── Step 6: Training loop ────────────────────────────────────────────────
    logger.info(f"Starting training: max_steps={max_steps}, grad_accum={grad_accum}")
    model.train()
    step        = 0
    accum_loss  = 0.0
    optimizer.zero_grad()

    while step < max_steps:
        for batch in loader:
            if step >= max_steps:
                break

            input_ids = batch.to(device)

            # Labels = input_ids shifted by 1 (causal LM objective)
            labels = input_ids.clone()
            labels[labels == pad_id] = -100   # ignore padding in loss

            with torch.cuda.amp.autocast(dtype=torch.float16):
                outputs = model(input_ids=input_ids, labels=labels)
                loss    = outputs.loss / grad_accum

            loss.backward()
            accum_loss += loss.item()

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                logger.info(f"Step {step + 1}/{max_steps} | loss: {accum_loss:.4f}")
                accum_loss = 0.0

            step += 1

    # ── Step 7: Save ─────────────────────────────────────────────────────────
    save_path = Path(train_cfg["output_dir"]) / "final_model"
    save_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving LoRA adapter to {save_path}...")
    model.save_pretrained(str(save_path))
    tokenizer.save_pretrained(str(save_path))

    logger.info("=" * 60)
    logger.info(f"Done. Adapter saved to {save_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
