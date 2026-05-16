"""
train_hf.py — Pure HuggingFace QLoRA training, zero Unsloth dependency.

Uses:  transformers + peft + bitsandbytes + torch (no Unsloth, no trl)

Run from project root:
    python src/train_hf.py
"""

import json
import logging
import sys
from pathlib import Path

import torch
import yaml

# ── Logging ──────────────────────────────────────────────────────────────────
Path("outputs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("outputs/train_hf.log", mode="w"),
    ],
)
logger = logging.getLogger(__name__)

CONFIG_PATH = Path("config/qlora_config.yaml")
DATA_PATH   = Path("data/raw/dataset.json")


def load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


# ── Dataset ──────────────────────────────────────────────────────────────────
from torch.utils.data import Dataset as TorchDataset

class ShareGPTDataset(TorchDataset):
    def __init__(self, json_path, tokenizer, max_length):
        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        self.examples = []
        for item in raw:
            messages = []
            for turn in item.get("conversations", []):
                role = "user" if turn["from"] == "human" else "assistant"
                messages.append({"role": role, "content": turn["value"]})
            if not messages:
                continue
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            enc = tokenizer(
                text, truncation=True, max_length=max_length, return_tensors="pt"
            )
            self.examples.append(enc["input_ids"].squeeze(0))

        logger.info(f"Dataset: {len(self.examples)} examples loaded")

    def __len__(self):  return len(self.examples)
    def __getitem__(self, i): return self.examples[i]


def collate(batch, pad_id):
    max_len = max(x.size(0) for x in batch)
    out = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    for i, seq in enumerate(batch):
        out[i, -seq.size(0):] = seq
    return out


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    logger.info("=" * 60)
    logger.info("  Pure HuggingFace QLoRA — No Unsloth")
    logger.info("=" * 60)

    cfg       = load_config()
    mcfg      = cfg["model"]
    lcfg      = cfg["lora"]
    tcfg      = cfg["training"]
    max_len   = mcfg["max_seq_length"]
    max_steps = tcfg["max_steps"]
    lr        = float(tcfg["learning_rate"])
    grad_accum= tcfg["gradient_accumulation_steps"]

    # ── GPU check ────────────────────────────────────────────────────────────
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available.")
    device = torch.device("cuda")
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Load tokenizer ───────────────────────────────────────────────────────
    logger.info("STEP 1: Loading tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(mcfg["base_model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info("STEP 1 OK")

    # ── Load model in 4-bit ──────────────────────────────────────────────────
    logger.info("STEP 2: Loading model in 4-bit (bitsandbytes)...")
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit              = True,
        bnb_4bit_quant_type       = "nf4",
        bnb_4bit_compute_dtype    = torch.float16,
        bnb_4bit_use_double_quant = True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        mcfg["base_model_name"],
        quantization_config = bnb_config,
        device_map          = "auto",
    )
    model.config.use_cache = False          # Required for gradient checkpointing
    model.enable_input_require_grads()      # Required for PEFT
    logger.info("STEP 2 OK")

    # ── Attach LoRA via PEFT ─────────────────────────────────────────────────
    logger.info("STEP 3: Attaching LoRA adapters via PEFT...")
    from peft import LoraConfig, get_peft_model, TaskType
    lora_config = LoraConfig(
        r              = lcfg["r"],
        lora_alpha     = lcfg["alpha"],
        target_modules = lcfg["target_modules"],
        lora_dropout   = lcfg["dropout"],
        bias           = "none",
        task_type      = TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    logger.info("STEP 3 OK")

    # ── Dataset ──────────────────────────────────────────────────────────────
    logger.info("STEP 4: Preparing dataset...")
    from torch.utils.data import DataLoader
    dataset = ShareGPTDataset(DATA_PATH, tokenizer, max_len)
    loader  = DataLoader(
        dataset,
        batch_size = 1,
        shuffle    = True,
        collate_fn = lambda b: collate(b, tokenizer.pad_token_id),
    )
    logger.info("STEP 4 OK")

    # ── Optimizer ────────────────────────────────────────────────────────────
    logger.info("STEP 5: Setting up optimizer...")
    import bitsandbytes as bnb
    optimizer = bnb.optim.PagedAdamW8bit(
        [p for p in model.parameters() if p.requires_grad],
        lr = lr,
    )
    logger.info("STEP 5 OK")

    # ── Scaler for FP16 ──────────────────────────────────────────────────────
    scaler = torch.cuda.amp.GradScaler()

    # ── Training loop ────────────────────────────────────────────────────────
    logger.info(f"STEP 6: Training for {max_steps} steps...")
    model.train()
    step       = 0
    accum_loss = 0.0
    optimizer.zero_grad()

    while step < max_steps:
        for batch in loader:
            if step >= max_steps:
                break

            input_ids = batch.to(device)
            labels    = input_ids.clone()
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
                logger.info(f"  step {step+1}/{max_steps} | loss: {accum_loss:.4f}")
                accum_loss = 0.0

            step += 1

    logger.info("STEP 6 OK — Training complete")

    # ── Save ─────────────────────────────────────────────────────────────────
    save_path = Path(tcfg["output_dir"]) / "final_model"
    save_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"STEP 7: Saving LoRA adapter to {save_path}...")
    model.save_pretrained(str(save_path))
    tokenizer.save_pretrained(str(save_path))
    logger.info("STEP 7 OK")

    logger.info("=" * 60)
    logger.info(f"  Done! Adapter saved to: {save_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
