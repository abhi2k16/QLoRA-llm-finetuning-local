"""
dataset.py - Dataset loading and chat-template tokenization wrapper.

Reads a ShareGPT-format JSON file, applies the model's built-in chat template
to each conversation, and returns HuggingFace Dataset objects whose single
column ("text") contains the fully formatted, ready-to-train strings.

ShareGPT format expected in data/raw/dataset.json:
[
  {
    "conversations": [
      {"from": "human", "value": "Your question here"},
      {"from": "gpt",   "value": "The model answer here"}
    ]
  },
  ...
]

Multi-turn conversations are supported.
"""

import json
import logging
from pathlib import Path

from datasets import Dataset, DatasetDict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


ROLE_MAP = {
    "human": "user",
    "gpt": "assistant",
    "system": "system",
}


class LocalDatasetWrapper:
    """
    Loads a local ShareGPT JSON file and converts every conversation into a
    single formatted string using the tokenizer's apply_chat_template method.

    Args:
        tokenizer: A HuggingFace tokenizer that has a chat template defined.
        max_seq_length: Sequences longer than this are skipped.
    """

    def __init__(self, tokenizer, max_seq_length: int = 512):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def prepare_dataset(self, json_path: str) -> Dataset:
        """Return a single training dataset with one 'text' column."""
        raw_data = self._read_raw_data(json_path)
        formatted = self._format_conversations(raw_data)
        self._validate_non_empty(formatted)

        dataset = Dataset.from_dict({"text": formatted})
        logger.info("Dataset ready: %s examples", dataset.num_rows)
        return dataset

    def prepare_splits(
        self,
        json_path: str,
        eval_ratio: float = 0.0,
        seed: int = 3407,
    ) -> DatasetDict:
        """
        Return train/eval splits. If eval_ratio <= 0, only the train split is returned.
        """
        raw_data = self._read_raw_data(json_path)
        formatted = self._format_conversations(raw_data)
        self._validate_non_empty(formatted)

        dataset = Dataset.from_dict({"text": formatted})
        if eval_ratio <= 0:
            logger.info("Evaluation disabled; using all %s examples for training", dataset.num_rows)
            return DatasetDict({"train": dataset})

        if dataset.num_rows < 2:
            raise ValueError(
                "Need at least 2 valid examples to create a train/eval split. "
                "Add more data or disable evaluation."
            )

        split_ratio = min(max(eval_ratio, 0.0), 0.5)
        split_dataset = dataset.train_test_split(test_size=split_ratio, seed=seed)
        logger.info(
            "Dataset split ready: train=%s eval=%s",
            split_dataset["train"].num_rows,
            split_dataset["test"].num_rows,
        )
        return DatasetDict(
            {"train": split_dataset["train"], "eval": split_dataset["test"]}
        )

    def validate_dataset_file(self, json_path: str) -> dict:
        """
        Validate the source file and return a compact summary without training.
        """
        raw_data = self._read_raw_data(json_path)
        formatted = self._format_conversations(raw_data)

        summary = {
            "raw_examples": len(raw_data),
            "valid_examples": len(formatted),
            "skipped_examples": len(raw_data) - len(formatted),
            "max_seq_length": self.max_seq_length,
        }
        logger.info("Validation summary: %s", summary)
        return summary

    def _read_raw_data(self, json_path: str) -> list[dict]:
        path = Path(json_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset not found at '{json_path}'. "
                "Create data/raw/dataset.json in ShareGPT format before training."
            )

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(
                "dataset.json must be a JSON array (list) of conversation objects."
            )

        logger.info("Loaded %s raw conversations from %s", len(data), json_path)
        return data

    def _validate_non_empty(self, formatted: list[str]) -> None:
        if not formatted:
            raise ValueError(
                "No training examples remain after filtering. "
                "Either lower max_seq_length or add more / shorter data."
            )

    def _format_conversations(self, raw_data: list[dict]) -> list[str]:
        """
        Convert each ShareGPT conversation into a single chat-templated string.
        Conversations that exceed max_seq_length are dropped with a warning.
        """
        formatted = []
        skipped = 0

        for idx, item in enumerate(raw_data):
            if "conversations" not in item:
                logger.warning("Item %s has no 'conversations' key - skipping.", idx)
                skipped += 1
                continue

            messages = []
            for turn in item["conversations"]:
                role = ROLE_MAP.get(turn.get("from", ""))
                value = turn.get("value")

                if role is None:
                    logger.warning(
                        "Unknown role '%s' in item %s - skipping turn.",
                        turn.get("from"),
                        idx,
                    )
                    continue

                if not isinstance(value, str) or not value.strip():
                    logger.warning("Empty or invalid turn value in item %s - skipping turn.", idx)
                    continue

                messages.append({"role": role, "content": value})

            if not messages:
                logger.warning("Item %s produced no valid turns - skipping.", idx)
                skipped += 1
                continue

            try:
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            except Exception as exc:
                logger.warning("Chat template failed for item %s: %s - skipping.", idx, exc)
                skipped += 1
                continue

            token_count = len(self.tokenizer.encode(text))
            if token_count > self.max_seq_length:
                logger.debug(
                    "Item %s has %s tokens > %s - skipping.",
                    idx,
                    token_count,
                    self.max_seq_length,
                )
                skipped += 1
                continue

            formatted.append(text)

        if skipped > 0:
            logger.warning("Skipped %s conversations (bad format or too long).", skipped)

        logger.info(
            "Kept %s conversations after length filtering (max_seq_length=%s)",
            len(formatted),
            self.max_seq_length,
        )
        return formatted

    def preview(self, json_path: str, n: int = 2) -> None:
        """Print the first n formatted examples for quick sanity-checking."""
        raw_data = self._read_raw_data(json_path)
        examples = self._format_conversations(raw_data[: n * 2])
        for i, example in enumerate(examples[:n], start=1):
            print("\n" + ("-" * 60))
            print(f"Example {i}:")
            print("-" * 60)
            print(example)
        print("\n" + ("-" * 60) + "\n")
