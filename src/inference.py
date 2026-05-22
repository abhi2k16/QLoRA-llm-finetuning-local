"""
inference.py — Run a single prompt against your trained LoRA adapter.

-This script allows you to run a single prompt against a trained LoRA adapter. 
-It loads the base model in 4-bit precision, attaches the LoRA adapter, 
 and generates a response based on the input prompt. The script uses the 
 same chat template as during training to format the prompt correctly.
-The script accepts command-line arguments for the prompt, adapter path, 
 configuration file, maximum new tokens, and temperature.

Usage:
    python src/inference.py --prompt "Explain LoRA in one paragraph."
    python src/inference.py --prompt "What is machine learning?" --max-new-tokens 200
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

CONFIG_PATH   = Path("config/qlora_config.yaml")
ADAPTER_PATH  = Path("outputs/final_model")


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments. Defaults are set for adapter path and config file, but the prompt is required.
    Returns:
        argparse.Namespace: The parsed arguments. 
    contains:
        - prompt (str): The input prompt text to generate a response for.
        - adapter_path (str): The path to the trained LoRA adapter directory.
        - config (str): The path to the YAML configuration file.
        - max_new_tokens (int): The maximum number of new tokens to generate.
        - temperature (float): The sampling temperature for generation.
    """
    parser = argparse.ArgumentParser(description="Run inference with a trained LoRA adapter.")
    parser.add_argument("--prompt",         required=True,  help="Input prompt text")
    parser.add_argument("--adapter-path",   default=str(ADAPTER_PATH))
    parser.add_argument("--config",         default=str(CONFIG_PATH))
    parser.add_argument("--max-new-tokens", type=int,   default=128)
    parser.add_argument("--temperature",    type=float, default=0.7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    adapter_path   = Path(args.adapter_path)
    base_model     = cfg["model"]["base_model_name"]
    max_seq_length = cfg["model"]["max_seq_length"]

    if not adapter_path.exists():
        raise FileNotFoundError(
            f"Adapter not found at '{adapter_path}'. "
            "Run python src/train.py first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Load tokenizer from adapter directory (it was saved there during training)
    logger.info("Loading tokenizer from %s", adapter_path)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model in 4-bit
    logger.info("Loading base model: %s", base_model)
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit           = True,
        bnb_4bit_quant_type    = "nf4",
        bnb_4bit_compute_dtype = torch.float16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config = bnb_cfg,
        device_map          = "auto",
    )

    # Attach the LoRA adapter
    logger.info("Loading LoRA adapter from %s", adapter_path)
    from peft import PeftModel
    model = PeftModel.from_pretrained(base, str(adapter_path))
    model.eval()

    # Format the prompt with the chat template
    messages = [{"role": "user", "content": args.prompt}]
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize          = False,
        add_generation_prompt = True,
    )

    inputs = tokenizer(
        input_text,
        return_tensors = "pt",
        truncation     = True,
        max_length     = max_seq_length,
    ).to(device)

    logger.info("Generating response...")
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.float16):
            output_ids = model.generate(
                **inputs,
                max_new_tokens = args.max_new_tokens,
                temperature    = args.temperature,
                do_sample      = args.temperature > 0,
                use_cache      = True,
                pad_token_id   = tokenizer.eos_token_id,
            )

    # Decode only the newly generated tokens
    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
    response   = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    print("\n" + "─" * 60)
    print(f"Prompt:   {args.prompt}")
    print("─" * 60)
    print(f"Response: {response}")
    print("─" * 60 + "\n")


if __name__ == "__main__":
    main()
