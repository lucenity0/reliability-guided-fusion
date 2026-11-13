"""Seed process-global RNGs and restrict SDPA; PyTorch stays optional on import."""

import os
import random

import numpy as np


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    """Seed Python, NumPy and (if installed) PyTorch on all devices.

    Call before model construction or CUDA initialization. Deterministic mode
    disables cuDNN benchmarking and requests deterministic torch algorithms;
    unsupported operations warn. Reproducibility is scoped to the same software
    and hardware, not across devices/releases. Independent NumPy Generators and
    DataLoader workers need their own seeds. Set PYTHONHASHSEED before launching
    Python if deterministic string hashing is needed; setting it here is too late.
    """
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if type(deterministic) is not bool:
        raise ValueError("deterministic must be a boolean")

    random.seed(seed)
    np.random.seed(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    try:
        import torch
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
        return

    # torch.manual_seed seeds CPU, CUDA and MPS RNGs.
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def restrict_sdpa(device, *, deterministic: bool) -> dict[str, bool]:
    """Restrict CUDA SDPA for deterministic runs and report the enabled backends.

    Call after seeding and device selection, before training or prediction. The
    restriction is process-global so it also covers validation. Other devices
    and non-deterministic runs leave existing flags untouched; this function
    does not undo restrictions from an earlier call in the same process.
    Returned flags describe backend eligibility, not which kernel ran.
    """
    import torch

    cuda = torch.backends.cuda
    if deterministic and torch.device(device).type == "cuda":
        cuda.enable_flash_sdp(False)
        cuda.enable_mem_efficient_sdp(False)
        cuda.enable_math_sdp(True)
    return {
        "flash": cuda.flash_sdp_enabled(),
        "mem_efficient": cuda.mem_efficient_sdp_enabled(),
        "math": cuda.math_sdp_enabled(),
    }
