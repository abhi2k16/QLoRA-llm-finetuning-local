# QLoRA Local Fine-Tuning

This project fine-tunes a small instruction model locally on Windows using Unsloth, HuggingFace, and QLoRA. It is designed for low-VRAM GPUs and keeps hyperparameters in `config/qlora_config.yaml`.

## Project Layout

```text
config/qlora_config.yaml   Training and evaluation settings
data/raw/dataset.json      ShareGPT-format training data
src/dataset.py             Dataset formatting, validation, and split logic
src/train.py               Main fine-tuning entrypoint
src/validate_dataset.py    Dataset validation and preview script
src/inference.py           Inference script for a saved adapter
outputs/                   Logs, checkpoints, and final adapter
```

## Environment Setup

Use Python 3.11 on Windows and install dependencies in this order:

```powershell
conda create -n win_llm python=3.11 -y
conda activate win_llm
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install "unsloth[windows] @ git+https://github.com/unslothai/unsloth.git"
pip install -r requirements.txt
```

If `bitsandbytes` fails from `requirements.txt`, install the Windows build manually:

```powershell
pip install bitsandbytes --prefer-binary --extra-index-url=https://jllllll.github.io/bitsandbytes-windows-webui
```

## Dataset Format

`data/raw/dataset.json` must be a JSON array in ShareGPT format:

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

## Validate the Dataset

Run this before training to catch formatting and length issues:

```powershell
python src/validate_dataset.py
```

## Train

The training script can now create a train/eval split and run periodic evaluation when enabled in the config.

```powershell
python src/train.py
```

## Inference

After training finishes, run inference against the saved adapter:

```powershell
python src/inference.py --prompt "Explain machine learning in simple terms."
```

## Evaluation Settings

`config/qlora_config.yaml` includes an `evaluation` section:

- Set `eval_ratio` to `0.0` to disable evaluation.
- Set it to something like `0.1` or `0.2` to hold out part of the dataset.
- `eval_steps` controls how often evaluation runs during training.

## Notes

- The sample dataset is only suitable for a smoke test.
- For real training, replace it with a larger task-specific dataset.
- This repo does not yet include automated tests for the ML pipeline.
