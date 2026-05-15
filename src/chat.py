"""
chat.py - Interactive local chat loop for a trained LoRA adapter.

Usage:
    python src/chat.py
    python src/chat.py --adapter-path outputs/final_model
"""

import argparse
from pathlib import Path

import torch
import yaml
from unsloth import FastLanguageModel


CONFIG_PATH = Path("config/qlora_config.yaml")


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--adapter-path", default="outputs/final_model")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--system-prompt", default="")
    return parser.parse_args()


def build_messages(system_prompt: str) -> list[dict]:
    messages = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    return messages


def generate_reply(model, tokenizer, messages: list[dict], max_new_tokens: int, temperature: float) -> str:
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
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    prompt_length = inputs.shape[-1]
    reply_tokens = outputs[0][prompt_length:]
    return tokenizer.decode(reply_tokens, skip_special_tokens=True).strip()


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.adapter_path,
        max_seq_length=cfg["model"]["max_seq_length"],
        load_in_4bit=True,
        dtype=None,
    )
    FastLanguageModel.for_inference(model)

    messages = build_messages(args.system_prompt)

    print("Interactive chat ready. Type 'exit' or 'quit' to stop.")
    if args.system_prompt.strip():
        print(f"System prompt: {args.system_prompt.strip()}")

    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Exiting chat.")
            break

        messages.append({"role": "user", "content": user_input})
        reply = generate_reply(
            model=model,
            tokenizer=tokenizer,
            messages=messages,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        messages.append({"role": "assistant", "content": reply})
        print(f"Assistant: {reply}")


if __name__ == "__main__":
    main()
