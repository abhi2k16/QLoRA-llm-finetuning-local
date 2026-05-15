"""
train.py - Main QLoRA fine-tuning pipeline.

Reads configuration from config/qlora_config.yaml, loads the model with
4-bit quantization, attaches LoRA adapters, tokenizes the dataset, and runs
training through Unsloth + HuggingFace trl.

Run from the project root:
    python src/train.py
"""

import logging
import sys
from pathlib import Path

import torch
import yaml
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel

sys.path.insert(0, str(Path(__file__).parent))
from dataset import LocalDatasetWrapper


OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(OUTPUTS_DIR / "train.log", mode="w"),
    ],
)
logger = logging.getLogger(__name__)


CONFIG_PATH = Path("config/qlora_config.yaml")
DATA_PATH = Path("data/raw/dataset.json")
SEED = 3407


def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found at '{path}'. "
            "Create config/qlora_config.yaml before running."
        )
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    logger.info("Config loaded from %s", path)
    return cfg


def check_gpu() -> None:
    if not torch.cuda.is_available():
        logger.warning(
            "CUDA is not available. Training will run on CPU and be extremely slow. "
            "Ensure your NVIDIA driver, CUDA stack, and GPU-enabled PyTorch install are correct."
        )
        return

    device = torch.cuda.get_device_name(0)
    total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    logger.info("GPU detected: %s | Total VRAM: %.2f GB", device, total_vram)

    if total_vram < 3.0:
        logger.warning(
            "GPU has only %.2f GB VRAM. This config targets at least 3 GB and may OOM.",
            total_vram,
        )


def load_model(cfg: dict):
    model_name = cfg["model"]["base_model_name"]
    max_seq_length = cfg["model"]["max_seq_length"]

    logger.info("Loading base model: %s", model_name)
    logger.info("Max sequence length: %s", max_seq_length)
    logger.info("4-bit quantization: enabled")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        dtype=None,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("pad_token set to eos_token")

    logger.info("Base model loaded successfully")
    return model, tokenizer


def attach_lora(model, cfg: dict):
    lora_cfg = cfg["lora"]

    logger.info(
        "Attaching LoRA adapters: r=%s, alpha=%s, dropout=%s",
        lora_cfg["r"],
        lora_cfg["alpha"],
        lora_cfg["dropout"],
    )
    logger.info("Target modules: %s", lora_cfg["target_modules"])

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_cfg["r"],
        target_modules=lora_cfg["target_modules"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        "Trainable parameters: %s / %s (%.2f%%)",
        f"{trainable:,}",
        f"{total:,}",
        100 * trainable / total,
    )
    return model


def build_training_args(cfg: dict, has_eval: bool) -> TrainingArguments:
    t = cfg["training"]
    eval_cfg = cfg.get("evaluation", {})

    Path(t["output_dir"]).mkdir(parents=True, exist_ok=True)

    evaluation_strategy = "steps" if has_eval else "no"
    eval_steps = eval_cfg.get("eval_steps", t["logging_steps"]) if has_eval else None

    args = TrainingArguments(
        per_device_train_batch_size=t["per_device_train_batch_size"],
        per_device_eval_batch_size=t.get("per_device_eval_batch_size", 1),
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        warmup_steps=t["warmup_steps"],
        max_steps=t["max_steps"],
        learning_rate=float(t["learning_rate"]),
        fp16=t["fp16"],
        bf16=t["bf16"],
        logging_steps=t["logging_steps"],
        output_dir=t["output_dir"],
        optim=t["optim"],
        report_to=t.get("report_to", "none"),
        save_strategy="steps",
        save_steps=t["max_steps"],
        save_total_limit=1,
        evaluation_strategy=evaluation_strategy,
        eval_steps=eval_steps,
        seed=SEED,
    )

    logger.info(
        "Training args: steps=%s, lr=%s, effective_batch=%s, optim=%s, eval=%s",
        t["max_steps"],
        t["learning_rate"],
        t["per_device_train_batch_size"] * t["gradient_accumulation_steps"],
        t["optim"],
        evaluation_strategy,
    )
    return args


def log_vram(label: str, peak: bool = False) -> None:
    if not torch.cuda.is_available():
        return

    if peak:
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)
        logger.info("Peak VRAM during %s: %.2f GB", label, peak_memory)
        return

    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    logger.info("VRAM %s - allocated: %.2f GB, reserved: %.2f GB", label, allocated, reserved)


def main() -> None:
    logger.info("=" * 60)
    logger.info("  Local QLoRA Fine-Tuning Pipeline")
    logger.info("=" * 60)

    check_gpu()
    cfg = load_config(CONFIG_PATH)
    model, tokenizer = load_model(cfg)
    model = attach_lora(model, cfg)

    logger.info("Loading dataset from %s", DATA_PATH)
    data_pipeline = LocalDatasetWrapper(
        tokenizer=tokenizer,
        max_seq_length=cfg["model"]["max_seq_length"],
    )

    eval_cfg = cfg.get("evaluation", {})
    dataset_splits = data_pipeline.prepare_splits(
        str(DATA_PATH),
        eval_ratio=float(eval_cfg.get("eval_ratio", 0.0)),
        seed=SEED,
    )
    train_dataset = dataset_splits["train"]
    eval_dataset = dataset_splits["eval"] if "eval" in dataset_splits else None

    training_args = build_training_args(cfg, has_eval=eval_dataset is not None)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=cfg["model"]["max_seq_length"],
        args=training_args,
        packing=False,
    )

    log_vram("before training")

    logger.info("Starting training...")
    train_result = trainer.train()
    logger.info("Training complete. Final loss: %.4f", train_result.training_loss)

    if eval_dataset is not None:
        logger.info("Running final evaluation on held-out split...")
        metrics = trainer.evaluate()
        logger.info("Evaluation metrics: %s", metrics)

    log_vram("training", peak=True)

    save_path = Path(cfg["training"]["output_dir"]) / "final_model"
    logger.info("Saving LoRA adapter to %s ...", save_path)
    model.save_pretrained_merged(str(save_path), tokenizer, save_method="lora")

    logger.info("=" * 60)
    logger.info("Done. Adapter saved to: %s", save_path)
    logger.info("Load it for inference with: python src/inference.py --prompt \"Your prompt\"")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
