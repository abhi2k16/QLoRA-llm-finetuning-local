# QLoRA Local Fine-Tuning

Fine-tune a small instruction model locally on Windows 11 using HuggingFace,
PEFT, and bitsandbytes. Designed for low-VRAM GPUs (3 GB). No Unsloth required.

## Stack

| Library | Role |
|---|---|
| `transformers` | Model loading, tokenisation |
| `peft` | LoRA adapter attachment |
| `bitsandbytes` | 4-bit quantization + paged optimizer |
| `torch` | Training loop, FP16 scaler |

## Environment setup

```powershell
# 1. Create conda environment
conda create -n win_llm python=3.11 -y
conda activate win_llm

# 2. PyTorch with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. bitsandbytes Windows build
pip install bitsandbytes --prefer-binary --extra-index-url https://jllllll.github.io/bitsandbytes-windows-webui

# 4. Remaining dependencies
pip install -r requirements.txt
```

## Dataset format

`data/raw/dataset.json` — ShareGPT JSON array:

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

## Usage

```powershell
# Validate dataset before training
python src/validate_dataset.py

# Train
python src/train.py

# Inference with saved adapter
python src/inference.py --prompt "Explain machine learning in simple terms."

# Interactive chat
python src/chat.py
python src/chat.py --system-prompt "You are a concise ML tutor."
```

## Config

All hyperparameters live in `config/qlora_config.yaml`. Key settings:

| Parameter | Default | Notes |
|---|---|---|
| `max_seq_length` | 512 | Reduce to 256 if OOM |
| `max_steps` | 60 | Increase to 300–500 for real training |
| `lora r` | 8 | Higher = more capacity, more VRAM |
| `learning_rate` | 0.0002 | Safe default for LoRA |

## Notes

- Unsloth is not used — it segfaults on Pascal GPUs (GTX 1050, compute 6.1)
- The sample dataset (4 examples) is for smoke-testing only
- Replace it with your own domain data for real fine-tuning
