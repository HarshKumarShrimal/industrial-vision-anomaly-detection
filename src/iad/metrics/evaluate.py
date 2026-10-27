"""Run a fitted model over a test split and compute all metrics + timing."""
from __future__ import annotations

import time
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from .auroc import compute_image_auroc, compute_pixel_auroc
from .pro import compute_pro


@torch.no_grad()
def evaluate(
    model,
    test_loader: DataLoader,
    compute_pro_metric: bool = True,
) -> Dict[str, float]:
    """Evaluate ``model`` on ``test_loader``.

    Returns a dict with image AUROC, pixel AUROC, PRO and average inference time
    per image (milliseconds). The model must already be fitted.
    """
    all_scores, all_labels = [], []
    all_maps, all_masks = [], []

    total_images = 0
    total_time = 0.0

    for batch in test_loader:
        images = batch["image"]
        start = time.perf_counter()
        pred = model.predict(images)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total_time += time.perf_counter() - start
        total_images += images.shape[0]

        all_scores.append(pred.image_scores.numpy())
        all_labels.append(np.asarray(batch["label"]))
        all_maps.append(pred.anomaly_maps.numpy())
        all_masks.append(batch["mask"].squeeze(1).numpy())

    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)
    maps = np.concatenate(all_maps, axis=0)
    masks = np.concatenate(all_masks, axis=0)

    results = {
        "image_auroc": compute_image_auroc(scores, labels),
        "pixel_auroc": compute_pixel_auroc(maps, masks),
        "inference_ms_per_image": 1000.0 * total_time / max(total_images, 1),
    }
    if compute_pro_metric:
        results["pro"] = compute_pro(maps, masks)
    return results
