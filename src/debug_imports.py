"""
debug_imports.py — Isolates which import in the non-Unsloth stack fails.

Run from project root:
    python src/debug_imports.py
"""

print("STEP 3a: importing transformers base...")
import transformers
print(f"  transformers version: {transformers.__version__}")
print("  STEP 3a OK")

print("STEP 3b: importing BitsAndBytesConfig...")
from transformers import BitsAndBytesConfig
print("  STEP 3b OK")

print("STEP 3c: importing AutoTokenizer...")
from transformers import AutoTokenizer
print("  STEP 3c OK")

print("STEP 3d: importing AutoModelForCausalLM...")
from transformers import AutoModelForCausalLM
print("  STEP 3d OK")

print("STEP 3e: importing peft...")
import peft
print(f"  peft version: {peft.__version__}")
print("  STEP 3e OK")

print("STEP 3f: importing LoRA helpers from peft...")
from peft import LoraConfig, get_peft_model
print("  STEP 3f OK")

print("\nAll imports OK for the non-Unsloth stack.")
print("Run python src/debug_train.py if the failure happens during model load or training.")
