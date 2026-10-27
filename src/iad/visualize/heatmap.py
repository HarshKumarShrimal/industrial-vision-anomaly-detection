"""Heatmap rendering: the visual "wow factor" for the README.

Produces the canonical 4-panel figure used throughout industrial AD papers:
original | ground-truth mask | predicted heatmap | overlay.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
_IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def denormalize(image: torch.Tensor) -> np.ndarray:
    """Undo ImageNet normalisation -> HxWx3 uint8-ready float in [0, 1]."""
    img = image.detach().cpu().numpy().transpose(1, 2, 0)
    img = img * _IMAGENET_STD + _IMAGENET_MEAN
    return np.clip(img, 0.0, 1.0)


def _normalize_map(amap: np.ndarray, vmin: Optional[float], vmax: Optional[float]) -> np.ndarray:
    vmin = float(amap.min()) if vmin is None else vmin
    vmax = float(amap.max()) if vmax is None else vmax
    if vmax - vmin < 1e-12:
        return np.zeros_like(amap)
    return np.clip((amap - vmin) / (vmax - vmin), 0.0, 1.0)


def overlay_heatmap(
    image: torch.Tensor,
    anomaly_map: np.ndarray,
    alpha: float = 0.5,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> np.ndarray:
    """Blend a jet-coloured anomaly map over the original image."""
    import matplotlib.cm as cm

    base = denormalize(image)
    norm = _normalize_map(np.asarray(anomaly_map), vmin, vmax)
    colored = cm.jet(norm)[..., :3]
    return np.clip((1 - alpha) * base + alpha * colored, 0.0, 1.0)


def save_comparison_grid(
    images: Sequence[torch.Tensor],
    anomaly_maps: Sequence[np.ndarray],
    masks: Sequence[torch.Tensor],
    out_path: str | Path,
    titles: Optional[Sequence[str]] = None,
    dpi: int = 150,
) -> Path:
    """Save a grid: each row = original | GT mask | heatmap | overlay.

    Args:
        images: list of normalised CxHxW tensors.
        anomaly_maps: list of HxW anomaly maps.
        masks: list of 1xHxW (or HxW) binary masks.
        out_path: where to write the PNG.
    """
    import matplotlib.pyplot as plt

    n = len(images)
    fig, axes = plt.subplots(n, 4, figsize=(4 * 3, n * 3))
    if n == 1:
        axes = axes[None, :]

    col_titles = ["Input", "Ground truth", "Anomaly heatmap", "Overlay"]
    for row in range(n):
        base = denormalize(images[row])
        amap = np.asarray(anomaly_maps[row])
        mask = masks[row]
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()
        mask = np.squeeze(mask)
        overlay = overlay_heatmap(images[row], amap)

        axes[row, 0].imshow(base)
        axes[row, 1].imshow(mask, cmap="gray", vmin=0, vmax=1)
        axes[row, 2].imshow(_normalize_map(amap, None, None), cmap="jet")
        axes[row, 3].imshow(overlay)

        for col in range(4):
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(col_titles[col], fontsize=12)
        if titles is not None:
            axes[row, 0].set_ylabel(titles[row], fontsize=10)

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path
