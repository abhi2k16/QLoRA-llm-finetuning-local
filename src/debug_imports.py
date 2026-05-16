"""
debug_imports.py — Isolates exactly which import line causes the segfault.

Run from project root:
    python src/debug_imports.py

Copy whichever STEP number is printed last before the crash and share it.
"""

print("STEP 3a: importing transformers base...")
import transformers
print(f"  transformers version: {transformers.__version__}")
print("  STEP 3a OK")

print("STEP 3b: importing TrainingArguments...")
from transformers import TrainingArguments
print("  STEP 3b OK")

print("STEP 3c: importing AutoTokenizer...")
from transformers import AutoTokenizer
print("  STEP 3c OK")

print("STEP 3d: importing AutoModelForCausalLM...")
from transformers import AutoModelForCausalLM
print("  STEP 3d OK")

print("STEP 3e: importing trl...")
import trl
print(f"  trl version: {trl.__version__}")
print("  STEP 3e OK")

print("STEP 3f: importing SFTTrainer from trl...")
from trl import SFTTrainer
print("  STEP 3f OK")

print("\nAll imports OK. The segfault is NOT at import time.")
print("Run python src/debug_train.py again — it must be crashing past STEP 3.")
