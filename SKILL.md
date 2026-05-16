---
name: local-llm-finetuning
description: >
  Fine-tune a small language model (SLM) on a low-VRAM GPU (3 GB) running Windows 11
  using pure HuggingFace + PEFT + bitsandbytes QLoRA. Use this skill whenever the
  user wants to fine-tune, train, or adapt an LLM on their own machine, mentions
  limited GPU memory or CUDA OOM errors, asks about QLoRA / LoRA adapters, wants
  to create a custom chatbot from their own dataset, or references SmolLM,
  bitsandbytes, or PEFT. Also trigger when the user asks how to run training
  locally, even if they don't say "fine-tune" explicitly.
---

# Local LLM Fine-Tuning Skill

Memory-safe QLoRA fine-tuning on Windows 11 with a 3 GB VRAM GPU.
Built on pure HuggingFace + PEFT + bitsandbytes — no Unsloth dependency.

---

## Hardware constraints

| Resource | Spec | Notes |
|---|---|---|
| GPU VRAM | 3 GB | GTX 1050 (Pascal / compute 6.1) or similar |
| Precision | FP16 only | BF16 not supported on Pascal architecture |
| CPU RAM | ≥ 16 GB | PagedAdamW spills optimizer state here |
| OS | Windows 11 | bitsandbytes Windows build required |
| CUDA | 12.1 | Install globally before any pip steps |

> **Why not Unsloth?** Unsloth's C/Triton kernels segfault on Pascal GPUs
> (compute 6.1). This project uses standard HuggingFace PEFT instead.

---

## Recommended models

- `HuggingFaceTB/SmolLM2-135M-Instruct` — primary, ~1.8 GB VRAM after quant
- `Qwen/Qwen2.5-0.5B-Instruct` — slightly stronger, ~2.4 GB VRAM after quant

---

## Project file structure

```
llm-finetuning-local/
├── SKILL.md
├── README.md
├── config/
│   └── qlora_config.yaml
├── data/
│   ├── raw/dataset.json
│   └── processed/
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── train.py
│   ├── validate_dataset.py
│   ├── inference.py
│   └── chat.py
├── outputs/
├── .gitignore
└── requirements.txt
```

---

## Key design decisions

**4-bit quantization** — `BitsAndBytesConfig(load_in_4bit=True)` cuts base model
VRAM by ~75%. Combined with LoRA (only adapter matrices are trained), the total
footprint stays under 3 GB.

**PagedAdamW8bit** — offloads optimizer state to CPU RAM and pages it in only when
needed. Essential on 3 GB VRAM.

**gradient_accumulation_steps: 4** — simulates effective batch of 4 using only
1 sample in VRAM at a time.

**FP16 + GradScaler** — GTX 1050 doesn't support BF16. GradScaler handles numeric
stability for FP16 training.

---

## Dataset format (ShareGPT)

```json
[
  {
    "conversations": [
      {"from": "human", "value": "What is LoRA?"},
      {"from": "gpt",   "value": "LoRA is a parameter-efficient fine-tuning method."}
    ]
  }
]
```

Multi-turn conversations are supported.

---

## Workflow

```powershell
# 1. Validate your dataset before training
python src/validate_dataset.py

# 2. Train
python src/train.py

# 3. Run inference on the saved adapter
python src/inference.py --prompt "Explain LoRA in simple terms."

# 4. Interactive chat
python src/chat.py
python src/chat.py --system-prompt "You are a concise ML tutor."
```

---

## Common errors and fixes

| Error | Cause | Fix |
|---|---|---|
| `CUDA out of memory` | Sequence too long | Reduce `max_seq_length` to 256 |
| Segfault on Unsloth import | Pascal GPU | Don't install Unsloth; not used here |
| bitsandbytes import error | Wrong Windows build | Reinstall from jllllll Windows index |
| `CUDA available: False` | CPU PyTorch build | Reinstall with `--index-url .../cu121` |
| SSL cert error | Broken env var | Set `SSL_CERT_FILE` to certifi cacert.pem |

---

## After training

Load the saved adapter for inference:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
base      = AutoModelForCausalLM.from_pretrained(
                "HuggingFaceTB/SmolLM2-135M-Instruct", quantization_config=bnb)
model     = PeftModel.from_pretrained(base, "outputs/final_model")
tokenizer = AutoTokenizer.from_pretrained("outputs/final_model")
```

Or use `src/inference.py` and `src/chat.py` which handle this automatically.

---

## Extending

- Raise `max_steps` to 300–500 for a real training run
- Replace `data/raw/dataset.json` with your domain-specific data
- Merge adapter: `model.merge_and_unload()` then `model.save_pretrained("outputs/merged")`
