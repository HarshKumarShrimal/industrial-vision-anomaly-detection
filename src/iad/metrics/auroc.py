"""AUROC metrics for detection (image-level) and localization (pixel-level)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def compute_image_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Image-level detection AUROC.

    Args:
        scores: ``(N,)`` anomaly score per image (higher = more anomalous).
        labels: ``(N,)`` binary labels (0=good, 1=defect).
    """
    labels = np.asarray(labels).astype(int)
    if labels.min() == labels.max():
        return float("nan")  # AUROC undefined with a single class.
    return float(roc_auc_score(labels, np.asarray(scores)))


def compute_pixel_auroc(maps: np.ndarray, masks: np.ndarray) -> float:
    """Pixel-level localization AUROC over all pixels of all test images.

    Args:
        maps: ``(N, H, W)`` anomaly maps.
        masks: ``(N, H, W)`` binary ground-truth masks.
    """
    flat_scores = np.asarray(maps).reshape(-1)
    flat_labels = (np.asarray(masks).reshape(-1) > 0.5).astype(int)
    if flat_labels.min() == flat_labels.max():
        return float("nan")
    return float(roc_auc_score(flat_labels, flat_scores))
