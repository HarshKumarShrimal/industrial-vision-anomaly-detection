"""High-level helpers that turn a config into a fitted model and metrics.

Keeps the CLI scripts and notebooks tiny: they just load a YAML and call
``run_experiment``.
"""
from __future__ import annotations

import time
from typing import Dict

import torch
from torch.utils.data import DataLoader

from .datasets.mvtec import MVTecDataset
from .metrics import evaluate
from .models.autoencoder import ConvAutoencoder
from .models.padim import PaDiM
from .models.patchcore import PatchCore
from .utils.config import Config
from .utils.device import resolve_device
from .utils.seed import set_seed

_METHODS = {"patchcore": PatchCore, "padim": PaDiM, "autoencoder": ConvAutoencoder}


def build_model(config: Config, device: str):
    """Instantiate a model from a config's ``method`` and ``model`` sections."""
    method = config["method"]
    if method not in _METHODS:
        raise ValueError(f"Unknown method '{method}'. Choose from {list(_METHODS)}.")
    params = dict(config.get("model", {}))
    params["device"] = device
    params["seed"] = config.get("seed", 42)
    params["image_size"] = config.get("data", {}).get("crop", 224)
    return _METHODS[method](**params)


def make_loaders(config: Config, category: str):
    data = config.get("data", {})
    root = data["root"]
    resize = data.get("resize", 256)
    crop = data.get("crop", 224)
    batch_size = data.get("batch_size", 8)
    workers = data.get("num_workers", 2)

    train_ds = MVTecDataset(root, category, "train", resize, crop)
    test_ds = MVTecDataset(root, category, "test", resize, crop)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=False, num_workers=workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=workers
    )
    return train_loader, test_loader


def run_experiment(config: Config, category: str, device: str | None = None) -> Dict[str, float]:
    """Fit on a category's good images and evaluate on its test split.

    Returns a metrics dict augmented with ``fit_seconds`` and the category name.
    """
    set_seed(config.get("seed", 42))
    device = str(resolve_device(device))

    train_loader, test_loader = make_loaders(config, category)
    model = build_model(config, device)

    t0 = time.perf_counter()
    model.fit(train_loader)
    fit_seconds = time.perf_counter() - t0

    metrics = evaluate(model, test_loader, compute_pro_metric=config.get("compute_pro", True))
    metrics["category"] = category
    metrics["method"] = config["method"]
    metrics["fit_seconds"] = fit_seconds
    return metrics
