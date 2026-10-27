"""Tests for AUROC and PRO metrics."""
from __future__ import annotations

import numpy as np

from iad.metrics import compute_image_auroc, compute_pixel_auroc, compute_pro


def test_image_auroc_perfect_separation():
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])
    assert compute_image_auroc(scores, labels) == 1.0


def test_image_auroc_random_is_half():
    scores = np.array([0.5, 0.5, 0.5, 0.5])
    labels = np.array([0, 1, 0, 1])
    assert abs(compute_image_auroc(scores, labels) - 0.5) < 1e-9


def test_pixel_auroc_perfect():
    maps = np.zeros((2, 8, 8))
    masks = np.zeros((2, 8, 8))
    maps[:, 2:5, 2:5] = 1.0
    masks[:, 2:5, 2:5] = 1.0
    assert compute_pixel_auroc(maps, masks) == 1.0


def test_pro_perfect_is_one():
    maps = np.zeros((3, 16, 16))
    masks = np.zeros((3, 16, 16))
    maps[:, 4:8, 4:8] = 1.0
    masks[:, 4:8, 4:8] = 1.0
    pro = compute_pro(maps, masks, num_thresholds=50)
    assert pro > 0.95


def test_pro_handles_multiple_regions_equally():
    # One image: a big region and a tiny region. A prediction that perfectly
    # covers both should score ~1 regardless of the size imbalance.
    maps = np.zeros((1, 32, 32))
    masks = np.zeros((1, 32, 32))
    masks[0, 0:10, 0:10] = 1   # big region
    masks[0, 20:22, 20:22] = 1  # tiny region
    maps[0] = masks[0].astype(float)
    pro = compute_pro(maps, masks, num_thresholds=50)
    assert pro > 0.9
