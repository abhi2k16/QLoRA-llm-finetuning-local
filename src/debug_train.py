"""
debug_train.py — Step-by-step isolation script for segmentation fault.

Run from project root:
    python src/debug_train.py

Each step prints a checkpoint before executing. The last printed checkpoint
before the crash tells you exactly which operation is causing the segfault.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

print("=" * 60)
print("STEP 1: Importing torch...")
import torch
print(f"  torch version : {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
print(f"  VRAM total    : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print("  STEP 1 OK")

print("\nSTEP 2: Importing bitsandbytes...")
import bitsandbytes as bnb
print(f"  bitsandbytes version: {bnb.__version__}")
print("  STEP 2 OK")

print("\nSTEP 3: Importing transformers + trl...")
from transformers import TrainingArguments
from trl import SFTTrainer
print("  STEP 3 OK")

print("\nSTEP 4: Importing Unsloth FastLanguageModel...")
from unsloth import FastLanguageModel
print("  STEP 4 OK")

print("\nSTEP 5: Loading base model with 4-bit quantization...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = "unsloth/SmolLM2-135M-Instruct",
    max_seq_length = 256,
    load_in_4bit   = True,
    dtype          = None,
)
print("  STEP 5 OK")

print("\nSTEP 6: Attaching LoRA adapters...")
model = FastLanguageModel.get_peft_model(
    model,
    r              = 8,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha     = 8,
    lora_dropout   = 0,
    bias           = "none",
    use_gradient_checkpointing = True,   # Using standard, not Unsloth custom
    random_state   = 3407,
)
print("  STEP 6 OK")

print("\nSTEP 7: Loading dataset...")
from datasets import Dataset
import json

with open("data/raw/dataset.json", "r") as f:
    raw = json.load(f)

texts = []
for item in raw:
    messages = []
    for turn in item["conversations"]:
        role = "user" if turn["from"] == "human" else "assistant"
        messages.append({"role": role, "content": turn["value"]})
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    texts.append(text)

dataset = Dataset.from_dict({"text": texts})
print(f"  Dataset rows: {dataset.num_rows}")
print("  STEP 7 OK")

print("\nSTEP 8: Building TrainingArguments...")
args = TrainingArguments(
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 2,
    warmup_steps                = 2,
    max_steps                   = 5,       # Only 5 steps — just enough to confirm no crash
    learning_rate               = 2e-4,
    fp16                        = True,
    bf16                        = False,
    logging_steps               = 1,
    output_dir                  = "outputs/debug",
    optim                       = "paged_adamw_8bit",
    report_to                   = "none",
    save_strategy               = "no",
)
print("  STEP 8 OK")

print("\nSTEP 9: Initialising SFTTrainer...")
trainer = SFTTrainer(
    model              = model,
    tokenizer          = tokenizer,
    train_dataset      = dataset,
    dataset_text_field = "text",
    max_seq_length     = 256,
    args               = args,
    packing            = False,
)
print("  STEP 9 OK")

print("\nSTEP 10: Logging VRAM before training...")
allocated = torch.cuda.memory_allocated() / 1e9
reserved  = torch.cuda.memory_reserved()  / 1e9
print(f"  Allocated: {allocated:.3f} GB")
print(f"  Reserved : {reserved:.3f} GB")
print("  STEP 10 OK")

print("\nSTEP 11: Running 5 training steps...")
print("  (This is the most likely crash point)")
result = trainer.train()
print(f"  Training loss: {result.training_loss:.4f}")
print("  STEP 11 OK")

print("\n" + "=" * 60)
print("ALL STEPS PASSED — segfault is resolved!")
print("You can now run: python src/train.py")
print("=" * 60)
