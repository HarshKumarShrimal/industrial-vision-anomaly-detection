"""Device resolution and deterministic seeding utilities."""
from __future__ import annotations

import random
import numpy as np
import torch


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    """Resolve compute device safely supporting cuda, mps, and cpu.

    If device is None or 'auto', prioritizes cuda -> mps -> cpu.
    Falls back gracefully if requested acceleration is not supported.
    """
    if isinstance(device, torch.device):
        return device

    if device is None or device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    dev_str = str(device).lower().strip()
    if dev_str.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    if dev_str == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        return torch.device("cpu")

    try:
        return torch.device(dev_str)
    except Exception:
        return torch.device("cpu")


def set_seed(seed: int = 42) -> None:
    """Set deterministic seeds across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
