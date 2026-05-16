"""
chat.py — Interactive multi-turn chat with your trained LoRA adapter.

Usage:
    python src/chat.py
    python src/chat.py --adapter-path outputs/final_model
    python src/chat.py --system-prompt "You are a concise ML tutor."

Type 'exit' or 'quit' to stop the session.
Type 'reset' to clear the conversation history and start fresh.
"""

import argparse
import logging
from pathlib import Path

import torch
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

CONFIG_PATH  = Path("config/qlora_config.yaml")
ADAPTER_PATH = Path("outputs/final_model")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive chat with a trained LoRA adapter.")
    parser.add_argument("--adapter-path",   default=str(ADAPTER_PATH))
    parser.add_argument("--config",         default=str(CONFIG_PATH))
    parser.add_argument("--max-new-tokens", type=int,   default=256)
    parser.add_argument("--temperature",    type=float, default=0.7)
    parser.add_argument("--system-prompt",  default="",
                        help="Optional system prompt to set the assistant's behaviour.")
    return parser.parse_args()


def load_model_and_tokenizer(base_model: str, adapter_path: str, device: torch.device):
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit           = True,
        bnb_4bit_quant_type    = "nf4",
        bnb_4bit_compute_dtype = torch.float16,
    )
    base  = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config = bnb_cfg,
        device_map          = "auto",
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, messages: list[dict], max_new_tokens: int,
             temperature: float, max_seq_length: int, device: torch.device) -> str:
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize              = False,
        add_generation_prompt = True,
    )
    inputs = tokenizer(
        input_text,
        return_tensors = "pt",
        truncation     = True,
        max_length     = max_seq_length,
    ).to(device)

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.float16):
            output_ids = model.generate(
                **inputs,
                max_new_tokens = max_new_tokens,
                temperature    = temperature,
                do_sample      = temperature > 0,
                use_cache      = True,
                pad_token_id   = tokenizer.eos_token_id,
            )

    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main() -> None:
    args = parse_args()

    adapter_path = Path(args.adapter_path)
    if not adapter_path.exists():
        raise FileNotFoundError(
            f"Adapter not found at '{adapter_path}'. "
            "Run python src/train.py first."
        )

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    base_model     = cfg["model"]["base_model_name"]
    max_seq_length = cfg["model"]["max_seq_length"]
    device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Loading model and adapter...")
    model, tokenizer = load_model_and_tokenizer(base_model, str(adapter_path), device)
    logger.info("Ready.")

    # Initialise conversation history
    def fresh_history() -> list[dict]:
        h = []
        if args.system_prompt.strip():
            h.append({"role": "system", "content": args.system_prompt.strip()})
        return h

    messages = fresh_history()

    print("\n" + "=" * 60)
    print("  Interactive Chat — type 'exit' to quit, 'reset' to restart")
    if args.system_prompt.strip():
        print(f"  System prompt: {args.system_prompt.strip()}")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        if user_input.lower() == "reset":
            messages = fresh_history()
            print("[Conversation reset]\n")
            continue

        messages.append({"role": "user", "content": user_input})

        reply = generate(
            model          = model,
            tokenizer      = tokenizer,
            messages       = messages,
            max_new_tokens = args.max_new_tokens,
            temperature    = args.temperature,
            max_seq_length = max_seq_length,
            device         = device,
        )

        messages.append({"role": "assistant", "content": reply})
        print(f"Assistant: {reply}\n")


if __name__ == "__main__":
    main()
