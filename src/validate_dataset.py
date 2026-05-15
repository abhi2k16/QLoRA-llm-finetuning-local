"""
validate_dataset.py - Validate a local ShareGPT dataset before training.

Usage:
    python src/validate_dataset.py
    python src/validate_dataset.py --dataset data/raw/dataset.json --max-seq-length 512
"""

import argparse
import json
import sys
from pathlib import Path

import yaml
from unsloth import FastLanguageModel

sys.path.insert(0, str(Path(__file__).parent))
from dataset import LocalDatasetWrapper


CONFIG_PATH = Path("config/qlora_config.yaml")


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--dataset", default="data/raw/dataset.json")
    parser.add_argument("--max-seq-length", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))

    model_name = cfg["model"]["base_model_name"]
    max_seq_length = args.max_seq_length or cfg["model"]["max_seq_length"]

    _, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        dtype=None,
    )

    wrapper = LocalDatasetWrapper(tokenizer=tokenizer, max_seq_length=max_seq_length)
    summary = wrapper.validate_dataset_file(args.dataset)
    print(json.dumps(summary, indent=2))
    wrapper.preview(args.dataset, n=2)


if __name__ == "__main__":
    main()
