---
name: local-llm-finetuning
description: >
  Fine-tune a small language model (SLM) on a low-VRAM GPU (3 GB) running Windows 11
  using Unsloth + HuggingFace + QLoRA. Use this skill whenever the user wants to
  fine-tune, train, or adapt an LLM on their own machine, mentions limited GPU memory
  or CUDA OOM errors, asks about QLoRA / LoRA adapters, wants to create a custom
  chatbot from their own dataset, or references Unsloth, SmolLM, or bitsandbytes.
  Also trigger when the user asks how to run training locally, even if they don't
  say "fine-tune" explicitly.
---

# Local LLM Fine-Tuning Skill

Guides Claude through setting up and executing a memory-safe QLoRA fine-tuning
pipeline on Windows 11 with a 3 GB VRAM GPU. Everything is engineered to prevent
CUDA Out-of-Memory (OOM) errors while producing a usable LoRA adapter.

---

## Hardware constraints

| Resource | Spec | Notes |
|---|---|---|
| GPU VRAM | 3 GB | GTX 1050 or similar Pascal/Turing card |
| Precision | FP16 only | BF16 is not supported on this architecture |
| CPU RAM | ≥ 16 GB | `paged_adamw_8bit` spills optimizer state here |
| OS | Windows 11 | Unsloth Windows wrapper required |
| CUDA | 12.1 | Must be installed globally before any pip steps |

---

## Recommended models

Both fit in ≤ 2.5 GB VRAM after 4-bit quantization, leaving a safe buffer.

- `unsloth/SmolLM2-135M-Instruct` — primary choice, smallest footprint
- `unsloth/Qwen2.5-0.5B-Instruct` — slightly richer reasoning, still safe

---

## Project file structure

```
llm-finetuning-local/
├── SKILL.md                    ← this file
├── config/
│   └── qlora_config.yaml       ← all hyperparams live here, not in code
├── data/
│   ├── raw/
│   │   └── dataset.json        ← ShareGPT format input
│   └── processed/              ← tokenised cache written here
├── src/
│   ├── __init__.py
│   ├── dataset.py              ← dataset loading & chat-template wrapper
│   └── train.py                ← full training pipeline
├── outputs/                    ← LoRA adapter weights saved here
├── .gitignore
└── requirements.txt
```

---

## Key design decisions

### Why Unsloth?
Unsloth provides custom Triton kernels that reduce VRAM overhead by up to 70%
compared to vanilla HuggingFace training. On a 3 GB card this is the difference
between OOM and a successful run.

### Why 4-bit quantization?
`load_in_4bit=True` compresses the base model weights by 75%. Combined with LoRA
(which only trains a tiny set of adapter matrices, not the full model), the total
VRAM footprint stays well within 3 GB.

### Why `paged_adamw_8bit`?
Standard Adam keeps optimizer state in VRAM. Paged AdamW 8-bit keeps it in CPU RAM
and pages it in only when needed — critical when VRAM is already almost full.

### Why `gradient_accumulation_steps: 4`?
With `per_device_train_batch_size: 1` the effective batch is just 1 sample. Accumulating
gradients over 4 forward passes before updating simulates a batch size of 4 without
storing 4 samples in VRAM simultaneously.

### Why `max_seq_length: 512`?
Attention memory scales quadratically with sequence length. Keeping it at 512 prevents
sudden VRAM spikes on long samples. Raise carefully if your data requires it.

---

## LoRA target modules

Always target all linear projection layers for SmolLM2 / Qwen2.5:

```
q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
```

---

## Dataset format (ShareGPT)

`data/raw/dataset.json` must be a JSON array of conversation objects:

```json
[
  {
    "conversations": [
      {"from": "human", "value": "What is the capital of France?"},
      {"from": "gpt",   "value": "Paris."}
    ]
  }
]
```

Each object is one training example. Multi-turn conversations are supported.

---

## Training pipeline summary

1. Load `qlora_config.yaml`
2. Load base model with 4-bit quantization via `FastLanguageModel.from_pretrained`
3. Attach LoRA adapters via `FastLanguageModel.get_peft_model`
4. Tokenise dataset via `LocalDatasetWrapper` (applies chat template)
5. Build `TrainingArguments` from config
6. Run `SFTTrainer.train()`
7. Save adapter with `model.save_pretrained_merged(..., save_method="lora")`

---

## Common errors and fixes

| Error | Likely cause | Fix |
|---|---|---|
| `CUDA out of memory` | Sequence too long or batch too large | Reduce `max_seq_length` to 256, confirm `batch_size=1` |
| `bf16 not supported` | Wrong precision flag | Set `fp16: true`, `bf16: false` in config |
| `ModuleNotFoundError: unsloth` | Wrong index URL or env mismatch | Confirm you're in `win_llm` conda env and installed from GitHub |
| `ValueError: dataset_text_field` | Wrong field name | Confirm `dataset_text_field = "text"` matches `LocalDatasetWrapper` output |
| Slow training (< 1 it/s) | Grad checkpointing overhead | Normal for 3 GB VRAM; 60 steps ≈ 2–5 min |

---

## After training

The adapter saved to `outputs/final_model` contains only the small LoRA delta —
not the full model. To run inference, load the base model and attach the adapter:

```python
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained("outputs/final_model")
FastLanguageModel.for_inference(model)
```

---

## Extending this setup

- Increase `max_steps` (e.g. 300–500) for a real training run once the fast test passes.
- Add a validation split to `LocalDatasetWrapper` for loss monitoring.
- Use `wandb` or `tensorboard` by adding `report_to` to `TrainingArguments`.
- Merge adapter into the base model for deployment:
  `save_method = "merged_16bit"` (requires ~4 GB disk, more RAM).
