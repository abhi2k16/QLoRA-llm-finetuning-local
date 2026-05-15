"""
train.py — Main QLoRA fine-tuning pipeline.

Reads all configuration from config/qlora_config.yaml, loads the model with
4-bit quantization, attaches LoRA adapters, tokenises the dataset, and runs
a memory-safe training loop via Unsloth + HuggingFace trl.

Run from the project root:
    python src/train.py

Ensure you are inside the 'win_llm' conda environment before running:
    conda activate win_llm
"""

import logging
import sys
from pathlib import Path

import torch
import yaml
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel

# Local import — must run from the project root, or add src/ to sys.path.
sys.path.insert(0, str(Path(__file__).parent))
from dataset import LocalDatasetWrapper

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("outputs/train.log", mode="w"),
    ],
)
logger = logging.getLogger(__name__)


# ─── Config path ─────────────────────────────────────────────────────────────
CONFIG_PATH = Path("config/qlora_config.yaml")
DATA_PATH   = Path("data/raw/dataset.json")


def load_config(path: Path) -> dict:
    """Load and return the YAML configuration file."""
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found at '{path}'. "
            "Create config/qlora_config.yaml before running."
        )
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    logger.info(f"Config loaded from {path}")
    return cfg


def check_gpu() -> None:
    """Log GPU availability and VRAM. Warn if CUDA is not available."""
    if not torch.cuda.is_available():
        logger.warning(
            "CUDA is not available! Training will run on CPU and be extremely slow. "
            "Ensure NVIDIA CUDA Toolkit 12.1 is installed and your GPU is detected."
        )
        return

    device = torch.cuda.get_device_name(0)
    total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    logger.info(f"GPU detected: {device} | Total VRAM: {total_vram:.2f} GB")

    if total_vram < 3.0:
        logger.warning(
            f"GPU has only {total_vram:.2f} GB VRAM. "
            "This config is designed for ≥3 GB. OOM errors are possible."
        )


def load_model(cfg: dict):
    """
    Load the base model with 4-bit quantization and return (model, tokenizer).

    4-bit quantization (load_in_4bit=True) compresses the base model weights
    by ~75%, bringing a 500M-parameter model from ~2 GB to ~0.5 GB VRAM.
    """
    model_name    = cfg["model"]["base_model_name"]
    max_seq_length = cfg["model"]["max_seq_length"]

    logger.info(f"Loading base model: {model_name}")
    logger.info(f"Max sequence length: {max_seq_length}")
    logger.info("4-bit quantization: enabled")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = model_name,
        max_seq_length = max_seq_length,
        load_in_4bit   = True,   # Compresses base model by 75%; essential for 3 GB VRAM
        dtype          = None,   # Auto-selects FP16 on GTX 1050 (no BF16 support)
    )

    # Ensure padding token is set (required by SFTTrainer for batching)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("pad_token set to eos_token")

    logger.info("Base model loaded successfully")
    return model, tokenizer


def attach_lora(model, cfg: dict):
    """
    Attach LoRA (Low-Rank Adaptation) adapters to the base model.

    LoRA freezes the base model and adds small trainable matrices to the
    target projection layers. Only these adapters are updated during training,
    keeping memory usage tiny compared to full fine-tuning.
    """
    lora_cfg = cfg["lora"]

    logger.info(
        f"Attaching LoRA adapters: r={lora_cfg['r']}, "
        f"alpha={lora_cfg['alpha']}, dropout={lora_cfg['dropout']}"
    )
    logger.info(f"Target modules: {lora_cfg['target_modules']}")

    model = FastLanguageModel.get_peft_model(
        model,
        r              = lora_cfg["r"],
        target_modules = lora_cfg["target_modules"],
        lora_alpha     = lora_cfg["alpha"],
        lora_dropout   = lora_cfg["dropout"],
        bias           = "none",

        # Unsloth's custom gradient checkpointing — reduces VRAM vs HuggingFace's
        # implementation. Required for 3 GB VRAM training.
        use_gradient_checkpointing = "unsloth",

        random_state   = 3407,   # Reproducibility seed
    )

    # Count trainable vs total parameters for logging
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Trainable parameters: {trainable:,} / {total:,} "
        f"({100 * trainable / total:.2f}%)"
    )

    return model


def build_training_args(cfg: dict) -> TrainingArguments:
    """Build HuggingFace TrainingArguments from the YAML config."""
    t = cfg["training"]

    Path(t["output_dir"]).mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        per_device_train_batch_size  = t["per_device_train_batch_size"],
        gradient_accumulation_steps  = t["gradient_accumulation_steps"],
        warmup_steps                 = t["warmup_steps"],
        max_steps                    = t["max_steps"],
        learning_rate                = float(t["learning_rate"]),
        fp16                         = t["fp16"],
        bf16                         = t["bf16"],
        logging_steps                = t["logging_steps"],
        output_dir                   = t["output_dir"],
        optim                        = t["optim"],
        report_to                    = t.get("report_to", "none"),
        save_steps                   = t["max_steps"],  # Save once at the end
        save_total_limit             = 1,
        seed                         = 3407,
    )

    logger.info(
        f"Training args: steps={t['max_steps']}, lr={t['learning_rate']}, "
        f"effective_batch={t['per_device_train_batch_size'] * t['gradient_accumulation_steps']}, "
        f"optim={t['optim']}"
    )
    return args


def main() -> None:
    logger.info("=" * 60)
    logger.info("  Local QLoRA Fine-Tuning — Memory-Optimised Pipeline")
    logger.info("=" * 60)

    # ── Step 1: Validate environment ─────────────────────────────────────────
    check_gpu()

    # ── Step 2: Load configuration ────────────────────────────────────────────
    cfg = load_config(CONFIG_PATH)

    # ── Step 3: Load base model with 4-bit quantization ───────────────────────
    model, tokenizer = load_model(cfg)

    # ── Step 4: Attach LoRA adapters ──────────────────────────────────────────
    model = attach_lora(model, cfg)

    # ── Step 5: Prepare dataset ───────────────────────────────────────────────
    logger.info(f"Loading dataset from {DATA_PATH}")
    data_pipeline     = LocalDatasetWrapper(
        tokenizer      = tokenizer,
        max_seq_length = cfg["model"]["max_seq_length"],
    )
    training_dataset  = data_pipeline.prepare_dataset(str(DATA_PATH))

    # ── Step 6: Build training arguments ──────────────────────────────────────
    training_args = build_training_args(cfg)

    # ── Step 7: Initialise SFTTrainer ─────────────────────────────────────────
    trainer = SFTTrainer(
        model             = model,
        tokenizer         = tokenizer,
        train_dataset     = training_dataset,
        dataset_text_field = "text",           # Column name from LocalDatasetWrapper
        max_seq_length    = cfg["model"]["max_seq_length"],
        args              = training_args,
        packing           = False,             # Keep False; packing can OOM on 3 GB
    )

    # ── Step 8: Log VRAM before training ──────────────────────────────────────
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved  = torch.cuda.memory_reserved()  / (1024 ** 3)
        logger.info(
            f"VRAM before training — allocated: {allocated:.2f} GB, "
            f"reserved: {reserved:.2f} GB"
        )

    # ── Step 9: Train ─────────────────────────────────────────────────────────
    logger.info("Starting training...")
    train_result = trainer.train()
    logger.info(f"Training complete. Final loss: {train_result.training_loss:.4f}")

    # ── Step 10: Log VRAM after training ──────────────────────────────────────
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
        logger.info(f"Peak VRAM during training: {peak:.2f} GB")

    # ── Step 11: Save LoRA adapter ────────────────────────────────────────────
    save_path = f"{cfg['training']['output_dir']}/final_model"
    logger.info(f"Saving LoRA adapter to {save_path} ...")

    # save_method="lora" saves only the small adapter delta, not the full model.
    # Use "merged_16bit" to merge adapter into base model (requires ~4 GB disk + RAM).
    model.save_pretrained_merged(save_path, tokenizer, save_method="lora")

    logger.info("=" * 60)
    logger.info(f"  Done! Adapter saved to: {save_path}")
    logger.info("  Load it for inference with:")
    logger.info(f"    from unsloth import FastLanguageModel")
    logger.info(f"    model, tokenizer = FastLanguageModel.from_pretrained('{save_path}')")
    logger.info(f"    FastLanguageModel.for_inference(model)")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
