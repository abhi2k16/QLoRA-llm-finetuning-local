"""
dataset.py — Dataset loading and chat-template tokenisation wrapper.

Reads a ShareGPT-format JSON file, applies the model's built-in chat template
to each conversation, and returns a HuggingFace Dataset object whose single
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

Multi-turn conversations (human → gpt → human → gpt → ...) are supported.
"""

import json
import logging
from pathlib import Path
from typing import Optional

from datasets import Dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ─── Role mapping ────────────────────────────────────────────────────────────
# ShareGPT uses "human" / "gpt"; the chat template expects "user" / "assistant".
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
        tokenizer: A HuggingFace tokenizer that has a chat template defined
                   (all Unsloth instruct models include one by default).
        max_seq_length: Sequences longer than this are silently skipped to
                        prevent out-of-memory spikes during training.
    """

    def __init__(self, tokenizer, max_seq_length: int = 512):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    # ── Public API ────────────────────────────────────────────────────────────

    def prepare_dataset(self, json_path: str) -> Dataset:
        """
        Main entry point. Reads the JSON file, formats every conversation,
        filters out over-length examples, and returns a Dataset.

        Args:
            json_path: Path to the ShareGPT JSON file.

        Returns:
            A HuggingFace Dataset with a single "text" column.
        """
        path = Path(json_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset not found at '{json_path}'. "
                "Create data/raw/dataset.json in ShareGPT format before training."
            )

        raw_data = self._load_json(path)
        logger.info(f"Loaded {len(raw_data)} raw conversations from {json_path}")

        formatted = self._format_conversations(raw_data)
        logger.info(f"Kept {len(formatted)} conversations after length filtering "
                    f"(max_seq_length={self.max_seq_length})")

        if len(formatted) == 0:
            raise ValueError(
                "No training examples remain after filtering. "
                "Either lower max_seq_length or add more / shorter data."
            )

        dataset = Dataset.from_dict({"text": formatted})
        logger.info(f"Dataset ready: {dataset.num_rows} examples")
        return dataset

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load_json(self, path: Path) -> list[dict]:
        """Read and validate the raw JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(
                "dataset.json must be a JSON array (list) of conversation objects."
            )
        return data

    def _format_conversations(self, raw_data: list[dict]) -> list[str]:
        """
        Convert each ShareGPT conversation into a single chat-templated string.
        Conversations that exceed max_seq_length are dropped with a warning.
        """
        formatted = []
        skipped = 0

        for idx, item in enumerate(raw_data):
            if "conversations" not in item:
                logger.warning(f"Item {idx} has no 'conversations' key — skipping.")
                skipped += 1
                continue

            # Convert ShareGPT roles → chat-template roles
            messages = []
            for turn in item["conversations"]:
                role = ROLE_MAP.get(turn.get("from", ""), None)
                if role is None:
                    logger.warning(
                        f"Unknown role '{turn.get('from')}' in item {idx} — skipping turn."
                    )
                    continue
                messages.append({"role": role, "content": turn["value"]})

            if not messages:
                logger.warning(f"Item {idx} produced no valid turns — skipping.")
                skipped += 1
                continue

            # Apply the model's chat template (adds special tokens, role tags, etc.)
            try:
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,          # Return a string, not token IDs
                    add_generation_prompt=False,
                )
            except Exception as e:
                logger.warning(f"Chat template failed for item {idx}: {e} — skipping.")
                skipped += 1
                continue

            # Length guard: tokenise and check length before adding to dataset
            token_count = len(self.tokenizer.encode(text))
            if token_count > self.max_seq_length:
                logger.debug(
                    f"Item {idx} has {token_count} tokens > {self.max_seq_length} — skipping."
                )
                skipped += 1
                continue

            formatted.append(text)

        if skipped > 0:
            logger.warning(f"Skipped {skipped} conversations (bad format or too long).")

        return formatted

    # ── Utility ───────────────────────────────────────────────────────────────

    def preview(self, json_path: str, n: int = 2) -> None:
        """
        Print the first n formatted examples. Useful for sanity-checking your
        dataset before launching a full training run.

        Usage:
            pipeline = LocalDatasetWrapper(tokenizer, max_seq_length=512)
            pipeline.preview("data/raw/dataset.json", n=3)
        """
        raw_data = self._load_json(Path(json_path))
        examples = self._format_conversations(raw_data[:n * 2])  # fetch extra in case some are skipped
        for i, example in enumerate(examples[:n]):
            print(f"\n{'─' * 60}")
            print(f"Example {i + 1}:")
            print(f"{'─' * 60}")
            print(example)
        print(f"\n{'─' * 60}\n")
