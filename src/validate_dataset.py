"""
validate_dataset.py — Validate your ShareGPT dataset before training.

Loads the tokenizer, applies the chat template to every conversation,
filters over-length examples, and prints a summary + two previews.

Usage:
    python src/validate_dataset.py
    python src/validate_dataset.py --dataset data/raw/dataset.json --max-seq-length 256
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))
from dataset import ShareGPTDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

CONFIG_PATH = Path("config/qlora_config.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a ShareGPT dataset.")
    parser.add_argument("--config",         default=str(CONFIG_PATH))
    parser.add_argument("--dataset",        default="data/raw/dataset.json")
    parser.add_argument("--max-seq-length", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    model_name     = cfg["model"]["base_model_name"]
    max_seq_length = args.max_seq_length or cfg["model"]["max_seq_length"]

    logger.info("Loading tokenizer: %s", model_name)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    wrapper = ShareGPTDataset(tokenizer=tokenizer, max_seq_length=max_seq_length)
    summary = wrapper.validate(args.dataset)

    print("\n" + "=" * 60)
    print("  Dataset Validation Summary")
    print("=" * 60)
    print(json.dumps(summary, indent=2))

    wrapper.preview(args.dataset, n=2)

    if summary["skipped"] > 0:
        logger.warning(
            "%d examples were skipped. Check your dataset format or lower --max-seq-length.",
            summary["skipped"],
        )
    else:
        logger.info("All examples passed validation. Ready to train.")


if __name__ == "__main__":
    main()
