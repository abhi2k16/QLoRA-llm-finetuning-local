# How I Fine-Tuned a Language Model on a 3 GB GPU — Without Losing My Mind

### A complete, battle-tested guide to QLoRA fine-tuning on Windows 11 using HuggingFace, PEFT, and bitsandbytes

---

> **TL;DR** — This guide walks you through fine-tuning a small language model
> on a low-end consumer GPU (3 GB VRAM, GTX 1050) on Windows 11. No cloud
> credits required. No expensive hardware. Just your machine, Python, and a
> dataset of your own conversations. We cover the theory, the setup, the code,
> the errors you *will* hit, and how to get past all of them.

---

## Who This Is For

You have a mid-range or older NVIDIA GPU sitting in your PC. You have heard
terms like "fine-tuning", "LoRA", and "QLoRA" thrown around in AI circles.
You want to actually *do* this — not just read about it — but every tutorial
you find assumes you have a cloud instance with 24 GB of VRAM or a budget for
a few hundred GPU-hours.

This guide is different. It is written from the actual experience of getting
this working on a GTX 1050 with 3 GB of VRAM on Windows 11. Every error shown
here was real. Every fix was tested. Nothing is handwaved.

By the end you will have a working fine-tuning pipeline, a trained LoRA
adapter, and an interactive chat script you can run against your own
custom model.

---

## Table of Contents

1. What Is Fine-Tuning and Why Does It Matter?
2. The Core Concepts You Need to Understand
3. System Requirements
4. Environment Setup — Step by Step
5. Project Structure
6. Understanding the Configuration File
7. Preparing Your Dataset
8. The Training Pipeline — Code Walkthrough
9. Running the Full Workflow
10. Errors I Hit and How I Fixed Them
11. Limitations of This Setup
12. Future Scope
13. Summary

---

## 1. What Is Fine-Tuning and Why Does It Matter?

A large language model like GPT or LLaMA is trained on hundreds of billions of
tokens scraped from the internet. It learns to predict the next word in a
sequence — and through doing this billions of times, it builds up a remarkable
general ability to understand and generate language.

But "general" is the key word. A base model does not know your company's
internal terminology, your writing style, your product domain, or how you want
it to behave in a conversation. It also does not follow instructions reliably
unless it has been specifically trained to do so.

**Fine-tuning** is the process of taking that pre-trained base model and
continuing to train it — but on a much smaller, targeted dataset that you
control. You are not training from scratch. You are nudging an already capable
model to specialise.

The challenge is that even a "small" language model has hundreds of millions of
parameters. Updating all of them requires enormous memory. A full fine-tune of
a 7B model needs roughly 40–80 GB of VRAM — way beyond what any consumer GPU
offers.

This is where **QLoRA** enters the picture.

---

## 2. The Core Concepts You Need to Understand

Before we touch any code, let's make sure you understand the four ideas that
make this whole project possible.

### 2.1 Quantization — Shrinking the Model

Modern language models store their weights as 32-bit or 16-bit floating point
numbers. Every number takes 4 bytes or 2 bytes of memory.

**Quantization** is the process of storing those numbers using fewer bits.
With **4-bit quantization**, each weight takes only 0.5 bytes. A model that
would normally consume 2 GB of VRAM suddenly needs only 500 MB.

The trade-off is precision — a 4-bit number cannot represent as wide a range
of values as a 16-bit one. The technique we use, called **NF4 (NormalFloat4)**,
is specifically designed to lose as little information as possible when
quantizing neural network weights, which tend to follow a bell-curve
distribution.

We use **bitsandbytes** to apply this quantization:

```python
from transformers import BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit              = True,
    bnb_4bit_quant_type       = "nf4",         # NormalFloat4 quantization
    bnb_4bit_compute_dtype    = torch.float16,  # Compute in FP16
    bnb_4bit_use_double_quant = True,           # Quantize the quantization constants too
)
```

The `double_quant` flag applies a second round of quantization to the scaling
constants themselves, saving an extra 0.4 bits per weight on average.

### 2.2 LoRA — Training Only a Tiny Slice

Even with the model quantized to 4 bits, we still cannot update all its weights.
The gradients and optimizer state for a full fine-tune would overflow our VRAM
many times over.

**LoRA (Low-Rank Adaptation)** solves this elegantly. Instead of updating the
original weight matrices, it freezes them completely and injects small, trainable
"adapter" matrices alongside the layers that matter most.

The mathematical insight is this: the *change* in a weight matrix during
fine-tuning tends to be low-rank — it lies in a much smaller subspace than the
full matrix. LoRA exploits this by decomposing the update into two small
matrices (A and B) whose product approximates the full update.

If a weight matrix W is shape 4096×4096 (16M parameters), a LoRA adapter with
rank 8 adds two matrices of shapes 4096×8 and 8×4096 — only 65K parameters.
That is a reduction of over 99%.

```python
from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    r              = 8,           # Rank — higher = more capacity but more VRAM
    lora_alpha     = 8,           # Scaling factor (keep equal to r)
    target_modules = [            # Which layers to attach adapters to
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout   = 0,
    bias           = "none",
    task_type      = TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
```

After training, you save only the adapter weights — not the full model. The
adapter is typically 5–50 MB, compared to 270 MB for the base model.

### 2.3 QLoRA — The Combination

**QLoRA** simply means doing LoRA on top of a 4-bit quantized model. The base
model sits in 4-bit precision in VRAM. The adapter matrices are trained in
16-bit precision. Only the adapter matrices receive gradient updates.

This combination is what makes fine-tuning feasible on consumer hardware.

### 2.4 Gradient Accumulation — Faking a Larger Batch

With only 3 GB of VRAM, we can only process one training sample at a time. But
training with a batch size of 1 is unstable — the gradient estimate from a
single example is noisy.

**Gradient accumulation** solves this by running several forward passes without
updating the weights, accumulating the gradients from each, and then doing a
single weight update at the end. With `gradient_accumulation_steps: 4`, we
simulate an effective batch size of 4 while only ever holding 1 sample in VRAM.

### 2.5 Paged Optimizer — Spilling to CPU RAM

The Adam optimizer keeps state for every trainable parameter — the first and
second moment estimates. Even for our small adapter, this state takes memory.

**PagedAdamW8bit** from bitsandbytes stores optimizer state in CPU RAM and
pages it into VRAM only when needed for a gradient update. Your 16 GB of system
RAM becomes an overflow buffer, preventing VRAM exhaustion.

### 2.6 FP16 + GradScaler — Mixed Precision Training

Our GPU (GTX 1050, Pascal architecture) supports FP16 arithmetic but not BF16.
Training entirely in FP16 risks numeric underflow — very small gradient values
round to zero and learning stops.

PyTorch's `GradScaler` solves this by scaling loss values up before the
backward pass, and scaling gradient values back down before the optimizer step.
This keeps gradients in a representable range throughout training.

---

## 3. System Requirements

### Minimum hardware

| Component | Minimum | Recommended |
|---|---|---|
| GPU | NVIDIA 3 GB VRAM (GTX 1050) | NVIDIA 6–8 GB VRAM |
| GPU Architecture | Pascal (compute 6.1) | Turing or Ampere |
| CPU RAM | 16 GB | 32 GB |
| Storage | 10 GB free | 20 GB free |
| OS | Windows 11 | Windows 11 |

### Software

| Software | Version | Notes |
|---|---|---|
| Python | 3.11 | Via Miniconda |
| CUDA Toolkit | 12.1 | Install globally before pip packages |
| NVIDIA Driver | ≥ 452.39 | Check with `nvidia-smi` |

### What this setup cannot do

- Fine-tune models larger than ~500M parameters (not enough VRAM)
- Use BF16 precision (Pascal architecture limitation)
- Run Unsloth (its Triton kernels segfault on compute 6.1)
- Train with batch sizes greater than 1 natively

---

## 4. Environment Setup — Step by Step

Open PowerShell and run these commands in exact order. The order matters.

**Step 1 — Create an isolated conda environment:**

```powershell
conda create -n win_llm python=3.11 -y
conda activate win_llm
```

**Step 2 — Install PyTorch with CUDA 12.1 support:**

This step is critical. If you run `pip install torch` without the index URL,
you will get a CPU-only build and nothing will work.

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Verify it worked:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
# Expected: 2.5.1+cu121 and True
```

**Step 3 — Install bitsandbytes using the Windows community build:**

The official PyPI build of bitsandbytes does not work reliably on Windows. Use
the community-maintained Windows build:

```powershell
pip install bitsandbytes --prefer-binary --extra-index-url https://jllllll.github.io/bitsandbytes-windows-webui
```

**Step 4 — Install the remaining dependencies:**

```powershell
pip install transformers>=4.40.0 datasets>=2.18.0 accelerate>=0.28.0 peft>=0.10.0 pyyaml certifi
```

**Step 5 — Fix SSL certificates (if HuggingFace downloads fail):**

Windows sometimes has a broken `SSL_CERT_FILE` environment variable that
prevents downloads. Fix it permanently:

```powershell
python -c "import certifi; print(certifi.where())"
# Copy the printed path, then:
[System.Environment]::SetEnvironmentVariable("SSL_CERT_FILE", "PASTE_PATH_HERE", "User")
```

Close and reopen PowerShell for this to take effect.

---

## 5. Project Structure

Here is how the project is organised:

```
QLoRA-llm-finetuning-local/
│
├── config/
│   └── qlora_config.yaml       ← All hyperparameters live here
│
├── data/
│   └── raw/
│       └── dataset.json        ← Your training conversations (ShareGPT format)
│
├── src/
│   ├── dataset.py              ← Loads and tokenises your dataset
│   ├── train.py                ← Main training pipeline
│   ├── validate_dataset.py     ← Sanity-check your data before training
│   ├── inference.py            ← Single-prompt inference
│   └── chat.py                 ← Interactive multi-turn chat
│
├── outputs/                    ← Adapter weights + logs written here
├── requirements.txt
└── README.md
```

The key design decision is that **all hyperparameters live in the YAML config
file, not in code**. This means you can experiment with different settings
without touching Python files, and your config history in Git tells the story
of how your training evolved.

---

## 6. Understanding the Configuration File

```yaml
model:
  base_model_name: "HuggingFaceTB/SmolLM2-135M-Instruct"
  max_seq_length: 512

lora:
  r: 8
  alpha: 8
  dropout: 0
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj

training:
  learning_rate: 0.0002
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 4
  warmup_steps: 5
  max_steps: 60
  fp16: true
  bf16: false
  optim: "paged_adamw_8bit"
  logging_steps: 1
  output_dir: "outputs"

evaluation:
  eval_ratio: 0.0
  eval_steps: 10
```

Let's go through the non-obvious settings:

**`max_seq_length: 512`** — Attention memory grows quadratically with sequence
length. At 512 tokens you are safe on 3 GB VRAM. If you push to 1024, you will
likely hit OOM. Reduce to 256 if you see memory errors.

**`r: 8` and `alpha: 8`** — The LoRA rank. Higher rank means more expressive
adapters but more VRAM and slower training. Keep rank equal to alpha as a
starting point. Do not exceed 16 on this hardware.

**`dropout: 0`** — Dropout during fine-tuning tends to hurt more than help on
small datasets. Leave it at 0.

**`max_steps: 60`** — This is a smoke-test value. It takes 2–5 minutes and
confirms the pipeline runs without errors. For real training, use 300–500 steps.

**`fp16: true` / `bf16: false`** — These must always be opposite. GTX 1050
supports FP16 but not BF16. Swapping these values will cause training to fail.

---

## 7. Preparing Your Dataset

The project expects your data in **ShareGPT format** — a JSON array of
conversation objects. Each object contains a list of turns between a human and
an AI assistant.

```json
[
  {
    "conversations": [
      {
        "from": "human",
        "value": "What is the difference between RAM and VRAM?"
      },
      {
        "from": "gpt",
        "value": "RAM is your computer's main memory, used by the CPU for
                  running applications. VRAM is dedicated memory on your GPU,
                  used specifically for graphics processing and, in our case,
                  storing model weights during training."
      }
    ]
  }
]
```

Multi-turn conversations are supported — you can have as many alternating
human/gpt turns as you like within a single conversation object.

### How much data do you need?

| Goal | Minimum examples | Recommended |
|---|---|---|
| Smoke test (does it run?) | 4–10 | — |
| Style/tone adaptation | 50–100 | 200+ |
| Domain knowledge injection | 200–500 | 1000+ |
| Behaviour change | 500+ | 2000+ |

The sample dataset in the repo has 4 examples — enough to confirm the
pipeline runs, not enough to actually change the model's behaviour meaningfully.

### Validate before training

Always run the validation script before kicking off a training run:

```powershell
python src/validate_dataset.py
```

This will apply the chat template, count tokens, filter over-length examples,
and print a summary and two previews. If the skipped count is non-zero,
investigate before training.

---

## 8. The Training Pipeline — Code Walkthrough

The training script (`src/train.py`) is structured as a series of clearly
separated functions. Let us walk through each one.

### Loading the model in 4-bit

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit              = True,
    bnb_4bit_quant_type       = "nf4",
    bnb_4bit_compute_dtype    = torch.float16,
    bnb_4bit_use_double_quant = True,
)

model = AutoModelForCausalLM.from_pretrained(
    "HuggingFaceTB/SmolLM2-135M-Instruct",
    quantization_config = bnb_config,
    device_map          = "auto",
)

model.config.use_cache = False      # Required for gradient checkpointing
model.enable_input_require_grads()  # Required for PEFT adapter attachment
```

The `device_map="auto"` tells HuggingFace to automatically place model layers
on the GPU. The two lines after are mandatory when using PEFT — without
`enable_input_require_grads()`, the adapter gradients will not flow.

### Attaching LoRA adapters

```python
from peft import LoraConfig, get_peft_model, TaskType

config = LoraConfig(
    r              = 8,
    lora_alpha     = 8,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_dropout   = 0,
    bias           = "none",
    task_type      = TaskType.CAUSAL_LM,
)
model = get_peft_model(model, config)
model.print_trainable_parameters()
# Output: trainable params: 819,200 || all params: 134,516,736 || trainable%: 0.609
```

We target all seven projection layers in the transformer. Missing any of them
reduces the adapter's capacity to learn. The trainable percentage will be well
under 1% — this is the whole point.

### The training loop

The core training loop avoids any high-level trainer abstraction and works
directly with PyTorch primitives:

```python
scaler    = torch.cuda.amp.GradScaler()   # FP16 stability
optimizer = bnb.optim.PagedAdamW8bit(
    [p for p in model.parameters() if p.requires_grad],
    lr = 0.0002,
)

model.train()
optimizer.zero_grad()

for step in range(max_steps):
    batch     = next(data_iterator)
    input_ids = batch.to(device)
    labels    = input_ids.clone()
    labels[labels == pad_token_id] = -100  # Ignore padding in loss

    with torch.cuda.amp.autocast(dtype=torch.float16):
        outputs = model(input_ids=input_ids, labels=labels)
        loss    = outputs.loss / gradient_accumulation_steps

    scaler.scale(loss).backward()

    if (step + 1) % gradient_accumulation_steps == 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        print(f"step {step+1} | loss: {loss.item():.4f}")
```

Four things to note here:

**1. Labels masking** — Setting `labels[labels == pad_token_id] = -100` tells
the model to compute loss only on real tokens, not padding. This is important
for correctness.

**2. Loss scaling** — We divide the loss by `gradient_accumulation_steps`
before the backward pass, so accumulated gradients represent the average over
the effective batch, not the sum.

**3. Gradient clipping** — `clip_grad_norm_` with a value of 1.0 prevents
exploding gradients, which are especially common in FP16 training.

**4. GradScaler** — The `scaler.scale()` / `scaler.unscale_()` / `scaler.step()`
/ `scaler.update()` sequence is the standard PyTorch pattern for FP16 mixed
precision. Never call `optimizer.step()` directly when using GradScaler.

### Saving the adapter

```python
save_path = Path("outputs/final_model")
save_path.mkdir(parents=True, exist_ok=True)

model.save_pretrained(str(save_path))       # Saves adapter weights only
tokenizer.save_pretrained(str(save_path))   # Saves tokenizer config
```

`model.save_pretrained()` on a PEFT model saves only the adapter delta — not
the full base model. The output directory will contain an `adapter_config.json`
and `adapter_model.safetensors` file, typically 3–15 MB total.

---

## 9. Running the Full Workflow

Here is the complete sequence from a fresh clone to an interactive chat session.

**Step 1 — Clone and set up:**

```powershell
git clone https://github.com/abhi2k16/QLoRA-llm-finetuning-local.git
cd QLoRA-llm-finetuning-local
conda activate win_llm
```

**Step 2 — Validate your dataset:**

```powershell
python src/validate_dataset.py
```

Expected output:

```
{
  "raw_examples": 4,
  "valid_examples": 4,
  "skipped_examples": 0,
  "max_seq_length": 512
}
```

**Step 3 — Run the smoke test (60 steps, ~3 minutes):**

```powershell
python src/train.py
```

Watch for the loss printed every step. It should trend downward. If it stays
flat or explodes, your learning rate or dataset likely needs adjustment.

```
2026-05-16 11:22:01 [INFO] step 4/60 | loss: 2.1043 | lr: 2.00e-04
2026-05-16 11:22:08 [INFO] step 8/60 | loss: 1.8821 | lr: 2.00e-04
2026-05-16 11:22:15 [INFO] step 12/60 | loss: 1.6204 | lr: 2.00e-04
...
2026-05-16 11:25:44 [INFO] step 60/60 | loss: 0.9871 | lr: 2.00e-04
2026-05-16 11:25:44 [INFO] Done! Adapter saved to: outputs/final_model
```

**Step 4 — Test with a single prompt:**

```powershell
python src/inference.py --prompt "What is the difference between LoRA and QLoRA?"
```

**Step 5 — Start an interactive chat:**

```powershell
python src/chat.py
# Or with a system prompt:
python src/chat.py --system-prompt "You are a concise machine learning tutor."
```

Type `reset` to clear the conversation history. Type `exit` to quit.

---

## 10. Errors I Hit and How I Fixed Them

This project involved a lot of debugging. Here is the honest record of every
significant error encountered, in the order they appeared.

### Error 1 — Unsloth cannot find any torch accelerator

```
NotImplementedError: Unsloth cannot find any torch accelerator? You need a GPU.
```

**Cause:** PyTorch was installed without CUDA support — the CPU-only build.

**Fix:**
```powershell
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Error 2 — xformers version conflict

```
ERROR: xformers 0.0.35 requires torch>=2.10, but you have torch 2.5.1+cu121
```

**Cause:** xformers was installed separately and requires a newer PyTorch.

**Fix:** This is a warning, not a fatal error. Just uninstall xformers:
```powershell
pip uninstall xformers -y
```

### Error 3 — torch.int1 AttributeError

```
AttributeError: module 'torch' has no attribute 'int1'
```

**Cause:** `torchao` was installed and requires PyTorch 2.6+, but we had 2.5.1.

**Fix:**
```powershell
pip uninstall torchao -y
```

### Error 4 — SSL certificate FileNotFoundError on HuggingFace download

```
FileNotFoundError: [Errno 2] No such file or directory
ssl.create_default_context(cafile=os.environ["SSL_CERT_FILE"])
```

**Cause:** The `SSL_CERT_FILE` environment variable pointed to a cert file that
no longer existed on disk.

**Fix:**
```powershell
pip install certifi
python -c "import certifi; print(certifi.where())"
# Set the printed path:
[System.Environment]::SetEnvironmentVariable("SSL_CERT_FILE", "<path>", "User")
```

### Error 5 — Segmentation fault on Unsloth import

```
🦥 Unsloth: Will patch your computer to enable 2x faster free finetuning.
Segmentation fault
```

**Cause:** Unsloth's compiled C/Triton kernels are not compatible with Pascal
GPU architecture (compute capability 6.1). This is a hard incompatibility —
not fixable through reinstallation or version changes.

**Fix:** Remove Unsloth entirely and migrate to pure HuggingFace + PEFT:
```powershell
pip uninstall unsloth unsloth-zoo trl xformers torchao -y
```

This was the biggest blocker in the project and required a full rewrite of the
training pipeline. The rewritten version does not use Unsloth at all.

---

## 11. Limitations of This Setup

Being honest about what this setup cannot do is as important as explaining
what it can.

### Model size ceiling

With 3 GB of VRAM, you are limited to models in the 100M–500M parameter range
after 4-bit quantization. Trying to load a 1B parameter model will OOM during
`from_pretrained()` before training even begins.

If you want to fine-tune a 7B or 13B model, you need at least 8–12 GB of VRAM.

### Training data volume

60–300 training steps on 4 example conversations will not meaningfully change
the model's behaviour. You will see the loss decrease, which means the model is
memorising your examples — but you need hundreds of diverse examples before
generalisation starts to happen.

Do not expect a 4-example fine-tune to produce a custom chatbot. It is a proof
of concept, not a production artefact.

### Training speed

On a GTX 1050, expect roughly 1–3 training steps per second. A 300-step run
takes around 5–10 minutes. A real training run of 2000 steps takes 30–60
minutes. This is slow by cloud GPU standards, but perfectly usable for
iterative experimentation.

### No BF16 support

Pascal GPUs (GTX 1050, 1060, 1070, 1080) do not support BF16 arithmetic.
BF16 has become the default precision in most modern fine-tuning frameworks
because it is more numerically stable than FP16 and requires no GradScaler.
We work around this with FP16 + GradScaler, but it adds complexity and
introduces occasional numeric instability on very long training runs.

### No Unsloth

Unsloth provides 1.5–2x training speed improvements through custom CUDA
kernels. Those kernels require compute capability 7.0 or higher (Turing+).
Our GTX 1050 (compute 6.1) is incompatible. We accept the slower training
speed in exchange for stability.

### Adapter, not merged model

The output of this pipeline is a LoRA *adapter* — not a standalone model.
To use it for inference, you need to load the base model and then load the
adapter on top. This is fine for local use, but if you want to deploy the
model as a standalone binary, you need to merge the adapter:

```python
merged = model.merge_and_unload()
merged.save_pretrained("outputs/merged_model")
```

Merging requires holding both the quantized base model and the adapter in
memory simultaneously, which works on 3 GB VRAM only for the smallest models.
For reliable merging, you need at least 6 GB.

---

## 12. Future Scope

This project is a foundation, not a finished product. Here is where it can go.

### Upgrade to a more capable GPU

The most impactful single change you can make is upgrading to a GPU with more
VRAM and a newer architecture. A GTX 1660 Ti (6 GB, Turing) unlocks BF16,
Unsloth support, larger models up to ~1B parameters, and significantly faster
training. An RTX 3060 (12 GB) opens up 3B–7B models.

### Add evaluation metrics

The current pipeline logs training loss, which tells you the model is learning
but not *what* it is learning or whether it generalises. A useful next step is
adding BLEU, ROUGE, or perplexity evaluation on a held-out validation split
to catch overfitting early.

### Multi-GPU training with Accelerate

HuggingFace's `accelerate` library (already in our dependencies) supports
distributed training across multiple GPUs with minimal code changes. If you
have two GPUs — even mismatched ones — you can distribute the model across
them and effectively double your VRAM budget.

### Experiment tracking

Uncomment `report_to: "wandb"` in the config to enable Weights & Biases
logging. This gives you a dashboard with loss curves, GPU utilisation, and
hyperparameter comparisons across runs — invaluable when you are tuning
`r`, `learning_rate`, and `max_steps` simultaneously.

### Larger and stronger base models

Once the pipeline is stable, try swapping `HuggingFaceTB/SmolLM2-135M-Instruct`
for `Qwen/Qwen2.5-0.5B-Instruct`. At 500M parameters it pushes closer to
the 3 GB VRAM ceiling but produces noticeably stronger responses. Further up,
`Qwen/Qwen2.5-1.5B-Instruct` requires ~6 GB VRAM but is a major quality step.

### Instruction dataset curation

The single highest-leverage improvement to fine-tuning quality is better data.
Curating 500–1000 high-quality, diverse conversations in your target domain
will improve model behaviour far more than any hyperparameter adjustment.
Tools like LlamaFactory and Axolotl provide structured workflows for dataset
curation alongside training.

### DPO / RLHF alignment

Once you have a supervised fine-tuned model, the next frontier is alignment —
teaching the model to prefer responses that are helpful, accurate, and
appropriately cautious. Direct Preference Optimisation (DPO) is the
most practical approach for local setups and requires preference data
(pairs of good/bad responses) rather than just conversation examples.

---

## 13. Summary

Here is everything covered in this guide, condensed to its essentials.

**The problem:** Fine-tuning language models is typically reserved for
high-end hardware. We made it work on a 3 GB consumer GPU.

**The techniques:**
- 4-bit NF4 quantization (bitsandbytes) to shrink the base model by 75%
- LoRA (PEFT) to train only a tiny fraction of parameters — under 1%
- PagedAdamW8bit to overflow optimizer state to CPU RAM
- Gradient accumulation to simulate larger batch sizes
- FP16 + GradScaler for numeric stability without BF16

**The stack:** HuggingFace `transformers` + `peft` + `bitsandbytes` + raw
PyTorch. No Unsloth — it segfaults on Pascal GPUs (compute 6.1).

**The workflow:**
```powershell
python src/validate_dataset.py   # Check your data
python src/train.py              # Train
python src/inference.py --prompt "Your question"
python src/chat.py               # Interactive session
```

**The limitations:** Small models only, slow training, FP16 only, adapter
output rather than standalone model, and a minimum of several hundred quality
examples for meaningful results.

**The upside:** It runs on hardware you already own, costs nothing, and gives
you complete control over your training data and model behaviour.

---

## Resources

- HuggingFace PEFT documentation: `huggingface.co/docs/peft`
- bitsandbytes documentation: `huggingface.co/docs/bitsandbytes`
- QLoRA paper (Dettmers et al., 2023): `arxiv.org/abs/2305.14314`
- LoRA paper (Hu et al., 2021): `arxiv.org/abs/2106.09685`
- Project repository: `github.com/abhi2k16/QLoRA-llm-finetuning-local`

---

*This guide was written from the ground up based on a real implementation
on a GTX 1050 running Windows 11. Every error shown was real. Every fix
was tested. If something does not work for you, open an issue on the GitHub
repo with your full error traceback and the output of
`python -c "import torch; print(torch.__version__, torch.cuda.is_available())"` —
that one line tells us most of what we need to diagnose the problem.*
