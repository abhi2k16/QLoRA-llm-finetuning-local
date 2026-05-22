"""
runtime_checks.py — Utility functions for runtime checks and error handling.  
Provides functions to check for CUDA availability and to verify that required Python modules are installed.
"""
from __future__ import annotations

import importlib

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


def require_module(module_name: str, install_hint: str, context: str):
    try:
        return importlib.import_module(module_name)
    except Exception as exc:
        raise RuntimeError(
            f"{context} requires the '{module_name}' package, but it could not be imported.\n"
            f"Install it with:\n  {install_hint}"
        ) from exc
