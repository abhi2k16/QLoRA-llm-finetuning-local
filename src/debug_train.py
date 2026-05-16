"""
debug_train.py — Step-by-step isolation script for the non-Unsloth stack.

Run from project root:
    python src/debug_train.py

Each step prints a checkpoint before executing. The last printed checkpoint
before the crash tells you which operation is failing.
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

print("\nSTEP 3: Importing transformers + peft...")
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, TaskType, get_peft_model
print("  STEP 3 OK")

print("\nSTEP 4: Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print("  STEP 4 OK")

print("\nSTEP 5: Loading base model with 4-bit quantization...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    "HuggingFaceTB/SmolLM2-135M-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
)
model.config.use_cache = False
model.enable_input_require_grads()
print("  STEP 5 OK")

print("\nSTEP 6: Attaching LoRA adapters...")
lora_config = LoraConfig(
    r=8,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=8,
    lora_dropout=0,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
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

print("\nSTEP 8: Building optimizer...")
optimizer = bnb.optim.PagedAdamW8bit(
    [p for p in model.parameters() if p.requires_grad],
    lr=2e-4,
)
print("  STEP 8 OK")

print("\nSTEP 9: Building a single debug batch...")
sample = tokenizer(
    dataset[0]["text"],
    truncation=True,
    max_length=256,
    return_tensors="pt",
)
input_ids = sample["input_ids"].to("cuda")
labels = input_ids.clone()
print("  STEP 9 OK")

print("\nSTEP 10: Logging VRAM before training...")
allocated = torch.cuda.memory_allocated() / 1e9
reserved  = torch.cuda.memory_reserved()  / 1e9
print(f"  Allocated: {allocated:.3f} GB")
print(f"  Reserved : {reserved:.3f} GB")
print("  STEP 10 OK")

print("\nSTEP 11: Running forward/backward/optimizer step...")
with torch.cuda.amp.autocast(dtype=torch.float16):
    outputs = model(input_ids=input_ids, labels=labels)
    loss = outputs.loss

loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
optimizer.zero_grad()
print(f"  Training loss: {loss.item():.4f}")
print("  STEP 11 OK")

print("\n" + "=" * 60)
print("ALL STEPS PASSED — the non-Unsloth training path is healthy.")
print("You can now run: python src/train.py")
print("=" * 60)
