"""Reproducibility helpers."""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Seed every RNG that the pipeline touches.

    Args:
        seed: Integer seed shared across ``random``, NumPy and PyTorch.
        deterministic: If True, force deterministic cuDNN kernels. This makes
            results reproducible at a small speed cost — exactly what we want
            for a benchmark.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
