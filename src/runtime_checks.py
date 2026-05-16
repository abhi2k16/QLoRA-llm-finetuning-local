from __future__ import annotations

import torch


CUDA_INSTALL_HINT = (
    "Install the CUDA 12.1 PyTorch wheel in your active environment:\n"
    "  python -m pip uninstall -y torch torchvision torchaudio\n"
    "  python -m pip install torch torchvision torchaudio "
    "--index-url https://download.pytorch.org/whl/cu121"
)


def require_cuda_runtime(context: str) -> None:
    if torch.cuda.is_available():
        return

    torch_version = getattr(torch, "__version__", "unknown")
    torch_cuda = getattr(torch.version, "cuda", None)

    raise RuntimeError(
        f"{context} requires a CUDA-enabled PyTorch install, but the active environment "
        f"has torch={torch_version} and torch.version.cuda={torch_cuda!r}.\n"
        f"{CUDA_INSTALL_HINT}"
    )


def get_fast_language_model(context: str):
    require_cuda_runtime(context)

    try:
        from unsloth import FastLanguageModel
    except Exception as exc:
        raise RuntimeError(
            f"{context} could not import Unsloth after CUDA was detected. "
            "Reinstall Unsloth in the same environment.\n"
            '  python -m pip install "unsloth[windows] @ git+https://github.com/unslothai/unsloth.git"'
        ) from exc

    return FastLanguageModel
