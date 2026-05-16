"""
dataset.py — ShareGPT dataset loader and chat-template tokenisation wrapper.

Reads a ShareGPT-format JSON file, applies the model's chat template to each
conversation, filters over-length examples, and returns HuggingFace Dataset
objects ready for training.

ShareGPT format (data/raw/dataset.json):
[
  {
    "conversations": [
      {"from": "human", "value": "Your question here"},
      {"from": "gpt",   "value": "The model answer here"}
    ]
  }
]

Multi-turn conversations are supported.
"""

import json
import logging
from pathlib import Path

from datasets import Dataset, DatasetDict

logger = logging.getLogger(__name__)

# ShareGPT role names → HuggingFace chat-template role names
ROLE_MAP = {
    "human":  "user",
    "gpt":    "assistant",
    "system": "system",
}


class ShareGPTDataset:
    """
    Loads a local ShareGPT JSON file, applies the tokenizer's chat template,
    and returns HuggingFace Dataset objects with a single "text" column.

    Args:
        tokenizer:      A HuggingFace tokenizer with a chat template defined.
        max_seq_length: Examples longer than this (in tokens) are skipped.
    """

    def __init__(self, tokenizer, max_seq_length: int = 512):
        self.tokenizer      = tokenizer
        self.max_seq_length = max_seq_length

    # ── Public API ────────────────────────────────────────────────────────────

    def prepare_dataset(self, json_path: str) -> Dataset:
        """Load, format, and return a single training Dataset."""
        raw  = self._load(json_path)
        text = self._format(raw)
        self._assert_non_empty(text, json_path)
        ds   = Dataset.from_dict({"text": text})
        logger.info("Dataset ready: %d examples", ds.num_rows)
        return ds

    def prepare_splits(
        self,
        json_path: str,
        eval_ratio: float = 0.0,
        seed: int = 42,
    ) -> DatasetDict:
        """
        Load and optionally split into train / eval.
        Returns a DatasetDict with keys "train" and optionally "eval".
        """
        raw  = self._load(json_path)
        text = self._format(raw)
        self._assert_non_empty(text, json_path)
        ds   = Dataset.from_dict({"text": text})

        if eval_ratio <= 0.0:
            logger.info("Evaluation disabled — using all %d examples for training", ds.num_rows)
            return DatasetDict({"train": ds})

        if ds.num_rows < 2:
            raise ValueError(
                "Need at least 2 examples for a train/eval split. "
                "Add more data or set eval_ratio: 0.0."
            )

        ratio  = min(max(eval_ratio, 0.0), 0.5)
        splits = ds.train_test_split(test_size=ratio, seed=seed)
        logger.info(
            "Split: train=%d  eval=%d",
            splits["train"].num_rows,
            splits["test"].num_rows,
        )
        return DatasetDict({"train": splits["train"], "eval": splits["test"]})

    def validate(self, json_path: str) -> dict:
        """Validate the file and return a summary dict without training."""
        raw    = self._load(json_path)
        text   = self._format(raw)
        summary = {
            "raw_examples":   len(raw),
            "valid_examples": len(text),
            "skipped":        len(raw) - len(text),
            "max_seq_length": self.max_seq_length,
        }
        logger.info("Validation summary: %s", summary)
        return summary

    def preview(self, json_path: str, n: int = 2) -> None:
        """Print the first n formatted examples for a quick sanity check."""
        raw      = self._load(json_path)
        examples = self._format(raw[: n * 2])
        for i, ex in enumerate(examples[:n], 1):
            print(f"\n{'─' * 60}\nExample {i}:\n{'─' * 60}")
            print(ex)
        print(f"\n{'─' * 60}\n")

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load(self, json_path: str) -> list[dict]:
        path = Path(json_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset not found at '{json_path}'. "
                "Create data/raw/dataset.json in ShareGPT format."
            )
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("dataset.json must be a JSON array of conversation objects.")
        logger.info("Loaded %d raw conversations from %s", len(data), json_path)
        return data

    def _format(self, raw: list[dict]) -> list[str]:
        """Apply chat template to each conversation; skip bad or over-length ones."""
        out     = []
        skipped = 0

        for idx, item in enumerate(raw):
            if "conversations" not in item:
                logger.warning("Item %d missing 'conversations' key — skipping.", idx)
                skipped += 1
                continue

            messages = []
            for turn in item["conversations"]:
                role  = ROLE_MAP.get(turn.get("from", ""))
                value = turn.get("value", "")
                if role is None:
                    logger.warning("Item %d: unknown role '%s' — skipping turn.", idx, turn.get("from"))
                    continue
                if not isinstance(value, str) or not value.strip():
                    logger.warning("Item %d: empty turn value — skipping turn.", idx)
                    continue
                messages.append({"role": role, "content": value})

            if not messages:
                logger.warning("Item %d: no valid turns — skipping.", idx)
                skipped += 1
                continue

            try:
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            except Exception as exc:
                logger.warning("Item %d: chat template error (%s) — skipping.", idx, exc)
                skipped += 1
                continue

            n_tokens = len(self.tokenizer.encode(text))
            if n_tokens > self.max_seq_length:
                logger.debug("Item %d: %d tokens > %d — skipping.", idx, n_tokens, self.max_seq_length)
                skipped += 1
                continue

            out.append(text)

        if skipped:
            logger.warning("Skipped %d conversations (bad format or too long).", skipped)
        logger.info(
            "Kept %d / %d conversations (max_seq_length=%d)",
            len(out), len(raw), self.max_seq_length,
        )
        return out

    def _assert_non_empty(self, text: list[str], json_path: str) -> None:
        if not text:
            raise ValueError(
                f"No valid training examples in '{json_path}'. "
                "Lower max_seq_length or add more / shorter data."
            )
