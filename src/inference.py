"""
inference.py - Run local inference with a trained LoRA adapter.

Usage:
    python src/inference.py --prompt "Explain LoRA in one paragraph."
"""

import argparse
from pathlib import Path

import torch
import yaml
from runtime_checks import get_fast_language_model


CONFIG_PATH = Path("config/qlora_config.yaml")


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--adapter-path", default="outputs/final_model")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))
    fast_language_model = get_fast_language_model("Inference")

    model, tokenizer = fast_language_model.from_pretrained(
        model_name=args.adapter_path,
        max_seq_length=cfg["model"]["max_seq_length"],
        load_in_4bit=True,
        dtype=None,
    )
    fast_language_model.for_inference(model)

    messages = [{"role": "user", "content": args.prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )

    if torch.cuda.is_available():
        inputs = inputs.to("cuda")

    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        do_sample=args.temperature > 0,
        use_cache=True,
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(decoded)


if __name__ == "__main__":
    main()
