---
name: local-llm-finetuning
description: >
  Fine-tune a small language model (SLM) on a low-VRAM GPU (3 GB) running Windows 11
  using Unsloth + HuggingFace + QLoRA. Use this skill whenever the user wants to
  fine-tune, train, validate data for, or run inference/chat with an LLM on their
  own machine, mentions limited GPU memory or CUDA OOM errors, asks about QLoRA /
  LoRA adapters, wants to create a custom chatbot from their own dataset, or
  references Unsloth, SmolLM, or bitsandbytes.
---

# Local LLM Fine-Tuning Skill

Guides the setup and execution of a memory-safe QLoRA fine-tuning pipeline on
Windows 11 with a 3 GB VRAM GPU. The project now includes dataset validation,
optional train/eval splitting, single-prompt inference, and an interactive chat
entrypoint for a trained adapter.

---

## Hardware constraints

| Resource | Spec | Notes |
|---|---|---|
| GPU VRAM | 3 GB | GTX 1050 or similar Pascal/Turing card |
| Precision | FP16 only | BF16 is not supported on this architecture |
| CPU RAM | >= 16 GB | `paged_adamw_8bit` spills optimizer state here |
| OS | Windows 11 | Unsloth Windows wrapper required |
| CUDA | 12.1 | Install before Python packages |

---

## Recommended models

Both fit within a low-VRAM workflow after 4-bit quantization.

- `unsloth/SmolLM2-135M-Instruct` - primary choice, smallest footprint
- `unsloth/Qwen2.5-0.5B-Instruct` - slightly stronger, still feasible

---

## Project file structure

```text
QLoRA-llm-finetuning-local/
|-- SKILL.md
|-- README.md
|-- config/
|   `-- qlora_config.yaml
|-- data/
|   `-- raw/
|       `-- dataset.json
|-- src/
|   |-- __init__.py
|   |-- dataset.py
|   |-- train.py
|   |-- validate_dataset.py
|   |-- inference.py
|   `-- chat.py
|-- outputs/
|-- .gitignore
`-- requirements.txt
```

---

## Key design decisions

### Why Unsloth?

Unsloth reduces memory overhead enough to make QLoRA practical on very small
consumer GPUs.

### Why 4-bit quantization?

`load_in_4bit=True` compresses the base weights and keeps the model inside the
VRAM budget when combined with LoRA adapters.

### Why `paged_adamw_8bit`?

It keeps optimizer state pressure off VRAM, which is necessary on 3 GB cards.

### Why `gradient_accumulation_steps: 4`?

It simulates a larger effective batch without storing multiple full examples in
VRAM at once.

### Why `max_seq_length: 512`?

Attention memory grows quickly with sequence length. `512` is a conservative
default for low-memory hardware.

---

## LoRA target modules

Target all major linear projection layers:

```text
q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
```

---

## Dataset format

`data/raw/dataset.json` must be a JSON array of ShareGPT-style conversation
objects:

```json
[
  {
    "conversations": [
      {"from": "human", "value": "What is LoRA?"},
      {"from": "gpt", "value": "LoRA is a parameter-efficient fine-tuning method."}
    ]
  }
]
```

Each object is one training example. Multi-turn conversations are supported.

---

## Current workflow

### 1. Validate the dataset

Run:

```powershell
python src/validate_dataset.py
```

This loads the configured tokenizer, checks ShareGPT formatting, applies the
chat template, filters over-length samples, prints a summary, and previews a few
examples.

### 2. Train with optional evaluation

Run:

```powershell
python src/train.py
```

The training pipeline:

1. Loads `config/qlora_config.yaml`
2. Loads the base model in 4-bit mode
3. Attaches LoRA adapters
4. Formats the dataset through `LocalDatasetWrapper`
5. Optionally creates a train/eval split
6. Builds `TrainingArguments`
7. Runs `SFTTrainer.train()`
8. Runs final evaluation if enabled
9. Saves the LoRA adapter to `outputs/final_model`

### 3. Run single-prompt inference

Run:

```powershell
python src/inference.py --prompt "Explain LoRA in simple terms."
```

### 4. Run interactive chat

Run:

```powershell
python src/chat.py
```

Optional:

```powershell
python src/chat.py --system-prompt "You are a concise ML tutor."
```

---

## Config behavior

`config/qlora_config.yaml` now contains:

- `model` settings for base model and `max_seq_length`
- `lora` settings for rank, alpha, dropout, and target modules
- `training` settings for batch size, optimizer, precision, logging, and outputs
- `evaluation` settings:
  - `eval_ratio`: hold-out fraction for validation
  - `eval_steps`: eval frequency during training

Set `eval_ratio: 0.0` to disable evaluation and train on the full dataset.

---

## Common errors and fixes

| Error | Likely cause | Fix |
|---|---|---|
| `CUDA out of memory` | Sequence too long or batch too large | Reduce `max_seq_length`, keep batch size at `1` |
| `bf16 not supported` | Wrong precision flags | Keep `fp16: true` and `bf16: false` |
| `ModuleNotFoundError: unsloth` | Environment mismatch | Activate `win_llm` and reinstall Unsloth |
| Dataset validation drops everything | Bad ShareGPT format or examples too long | Fix `dataset.json` or lower `max_seq_length` |
| No eval split created | Too little data or `eval_ratio: 0.0` | Add more examples or disable evaluation intentionally |

---

## After training

The adapter is saved to `outputs/final_model`. This is the LoRA delta, not a full
merged base model checkpoint.

Use:

- `python src/inference.py --prompt "..."`
- `python src/chat.py`

to interact with it.

---

## Extending this setup

- Increase `max_steps` once the smoke test passes
- Replace the sample dataset with real domain-specific conversations
- Enable `wandb` or `tensorboard` through `report_to`
- Save a merged model if deployment requires a standalone checkpoint
