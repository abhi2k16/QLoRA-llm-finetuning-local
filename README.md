# QLoRA Local Fine-Tuning

Local QLoRA fine-tuning project for Windows 11 with a low-VRAM NVIDIA GPU.
The stack is based on Hugging Face `transformers`, `peft`, `bitsandbytes`, and `torch`.
`unsloth` is intentionally not used in this project because it is unstable in this hardware setup.

## Overview

This repository is built for:

- Windows 11
- Python 3.11
- CUDA 12.1
- Low-VRAM GPUs such as GTX 1050 3 GB
- ShareGPT-style conversation datasets

The project supports:

- dataset validation before training
- 4-bit model loading with bitsandbytes
- LoRA adapter training
- optional train/eval split
- single-prompt inference
- interactive multi-turn chat
- debug scripts for imports and training-path isolation

## Tech Stack

| Library | Purpose |
|---|---|
| `transformers` | Base model, tokenizer, generation |
| `peft` | LoRA adapter attachment and loading |
| `bitsandbytes` | 4-bit quantization and paged 8-bit AdamW |
| `torch` | Core training loop, mixed precision, CUDA |
| `datasets` | Train/eval dataset containers and splitting |
| `pyyaml` | YAML config loading |

## Project Structure

```text
QLoRA-llm-finetuning-local/
├── config/
│   └── qlora_config.yaml        # Main training and model configuration
├── data/
│   └── raw/
│       └── dataset.json         # ShareGPT-format training data
├── outputs/                     # Logs, checkpoints, trained adapter output
│   └── final_model/             # Saved LoRA adapter and tokenizer files
├── src/
│   ├── __init__.py
│   ├── chat.py                  # Interactive chat with the trained adapter
│   ├── dataset.py               # ShareGPT loader, formatter, validator, splitter
│   ├── debug_imports.py         # Import-level debugging for the current stack
│   ├── debug_train.py           # Step-by-step training-path debug script
│   ├── inference.py             # Single prompt inference script
│   ├── runtime_checks.py        # CUDA and dependency runtime helpers
│   ├── train.py                 # Main training entrypoint
│   ├── train_hf.py              # Alternate pure Hugging Face training script
│   ├── train_raw.py             # Raw PyTorch training loop without Trainer/trl
│   └── validate_dataset.py      # Dataset validation and preview tool
├── .gitignore
├── README.md
├── requirements.txt
└── SKILL.md
```

## What Each Script Does

### `src/train.py`

Primary training entrypoint.

- loads config from `config/qlora_config.yaml`
- checks CUDA availability
- loads tokenizer and base model in 4-bit
- attaches LoRA adapters with PEFT
- builds a train-only or train/eval split from `data/raw/dataset.json`
- trains with a manual PyTorch loop and paged 8-bit AdamW
- saves the adapter and tokenizer under `outputs/final_model`

Use this first for the normal project workflow.

### `src/train_raw.py`

Simplified raw PyTorch training path.

- avoids `trl`
- avoids `unsloth`
- uses a direct manual loop over tokenized batches
- saves a standard PEFT adapter to `outputs/final_model`

Use this if you want the most direct training path with minimal moving parts.

### `src/train_hf.py`

Alternate pure Hugging Face QLoRA training script.

- still avoids `unsloth`
- keeps the training path separate from the main script
- useful as a fallback or comparison path

### `src/validate_dataset.py`

Dataset sanity checker.

- loads tokenizer from the configured base model
- applies the model chat template
- filters invalid or over-length examples
- prints a validation summary
- previews sample formatted conversations

Run this before training whenever the dataset changes.

### `src/dataset.py`

Reusable ShareGPT dataset utility.

- loads local JSON data
- maps `human` / `gpt` / `system` roles to chat-template roles
- validates structure
- skips malformed or too-long conversations
- returns Hugging Face `Dataset` or `DatasetDict`

### `src/inference.py`

Single-prompt generation using the trained LoRA adapter.

- loads base model in 4-bit
- loads adapter from `outputs/final_model`
- formats the prompt with the tokenizer chat template
- prints only the generated assistant response

### `src/chat.py`

Interactive CLI chat loop.

- loads the trained adapter
- maintains conversation history
- supports an optional system prompt
- supports `reset`, `exit`, and `quit`

### `src/runtime_checks.py`

Runtime helper functions.

- checks whether CUDA-enabled PyTorch is installed
- provides a generic import helper for required modules

### `src/debug_imports.py`

Quick import isolation script.

- checks whether the active environment can import the current stack cleanly
- useful when failures happen before model loading

### `src/debug_train.py`

Step-by-step debug script for the non-Unsloth training stack.

- imports each dependency stage
- loads tokenizer and base model
- attaches LoRA
- loads one sample batch
- runs a single forward/backward/optimizer step

Use this when the normal training path crashes and you want the failing stage.

## Environment Setup

Use the commands below in PowerShell.

```powershell
# 1. Create environment
conda create -n win_llm python=3.11 -y
conda activate win_llm

# 2. Install CUDA-enabled PyTorch first
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Install bitsandbytes Windows build
pip install bitsandbytes --prefer-binary --extra-index-url https://jllllll.github.io/bitsandbytes-windows-webui

# 4. Install remaining Python dependencies
pip install -r requirements.txt
```

If Hugging Face downloads fail because of SSL on Windows:

```powershell
pip install certifi
python -c "import certifi; print(certifi.where())"
# Copy the printed path, then:
$env:SSL_CERT_FILE = "PASTE_CERT_PATH_HERE"
```

## Dataset Format

The training file is expected at `data/raw/dataset.json`.

It must be a JSON array in ShareGPT-style format:

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

Supported `from` values in the current loader:

- `human`
- `gpt`
- `system`

Multi-turn conversations are supported.

## Standard Workflow

### 1. Validate the dataset

```powershell
python src/validate_dataset.py
```

Optional custom dataset path or sequence length:

```powershell
python src/validate_dataset.py --dataset data/raw/dataset.json --max-seq-length 256
```

### 2. Train the adapter

Primary path:

```powershell
python src/train.py
```

Fallback raw path:

```powershell
python src/train_raw.py
```

Alternate HF path:

```powershell
python src/train_hf.py
```

### 3. Run inference

```powershell
python src/inference.py --prompt "Explain machine learning in simple terms."
```

Example with custom length:

```powershell
python src/inference.py --prompt "What is QLoRA?" --max-new-tokens 200
```

### 4. Start interactive chat

```powershell
python src/chat.py
python src/chat.py --system-prompt "You are a concise ML tutor."
```

## Configuration

All main settings live in `config/qlora_config.yaml`.

### Model section

- `base_model_name`: base instruction model to fine-tune
- `max_seq_length`: maximum token length per conversation

Default configured model:

- `HuggingFaceTB/SmolLM2-135M-Instruct`

Commented alternative already shown in config:

- `Qwen/Qwen2.5-0.5B-Instruct`

### LoRA section

- `r`: LoRA rank
- `alpha`: LoRA scaling factor
- `dropout`: LoRA dropout
- `target_modules`: transformer linear layers that receive adapters

### Training section

- `learning_rate`
- `per_device_train_batch_size`
- `gradient_accumulation_steps`
- `warmup_steps`
- `max_steps`
- `fp16`
- `bf16`
- `optim`
- `logging_steps`
- `output_dir`
- `report_to`

### Evaluation section

- `eval_ratio`: hold-out fraction for validation
- `eval_steps`: reserved config value for evaluation cadence

## Current Default Config Notes

The config is tuned for a very small GPU.

| Setting | Current value | Why |
|---|---|---|
| `base_model_name` | `HuggingFaceTB/SmolLM2-135M-Instruct` | Small enough for low VRAM |
| `max_seq_length` | `512` | Safe starting point for 3 GB |
| `r` | `8` | Conservative LoRA size |
| `gradient_accumulation_steps` | `4` | Simulates larger batch size |
| `max_steps` | `60` | Smoke-test scale training |
| `fp16` | `true` | GTX 1050 supports FP16 |
| `bf16` | `false` | GTX 1050 does not support BF16 |

For a more stable first run, reduce `max_seq_length` from `512` to `256` if you hit OOM.

## Outputs

Training writes into `outputs/`.

Typical files and folders:

- `outputs/train.log`
- `outputs/train_raw.log`
- `outputs/train_hf.log`
- `outputs/final_model/adapter_config.json`
- `outputs/final_model/adapter_model.safetensors` or equivalent adapter weights
- `outputs/final_model/tokenizer.json`
- `outputs/final_model/tokenizer_config.json`
- `outputs/final_model/special_tokens_map.json`
- `outputs/final_model/merges.txt`

The exact tokenizer files depend on the base model tokenizer.

## Troubleshooting

### CUDA not available

Reinstall CUDA-enabled PyTorch:

```powershell
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Out of memory

Try the following:

- reduce `max_seq_length` from `512` to `256`
- keep `per_device_train_batch_size` at `1`
- keep `r` at `8`
- use `src/train_raw.py` if you want the simplest path

### Import or environment failures

Use:

```powershell
python src/debug_imports.py
```

### Training path crashes

Use:

```powershell
python src/debug_train.py
```

This will show the exact step that fails.

## Important Notes

- `unsloth` is not part of the active project path
- `trl` is not required for the current workflow
- the sample dataset is only for smoke testing
- real fine-tuning quality depends mostly on better domain data
- this repository saves LoRA adapters, not a fully merged standalone model
