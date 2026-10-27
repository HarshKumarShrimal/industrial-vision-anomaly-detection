"""Tests for the MVTec dataset loader."""
from __future__ import annotations

import torch

from iad.datasets.mvtec import MVTEC_CATEGORIES, MVTecDataset


def test_categories_count():
    assert len(MVTEC_CATEGORIES) == 15
    assert "bottle" in MVTEC_CATEGORIES


def test_train_split_is_all_good(synthetic_root):
    ds = MVTecDataset(synthetic_root, "bottle", "train", resize=64, crop=64)
    assert len(ds) == 6
    assert all(lbl == 0 for lbl in ds.labels)


def test_test_split_labels_and_masks(synthetic_root):
    ds = MVTecDataset(synthetic_root, "bottle", "test", resize=64, crop=64)
    assert len(ds) == 6  # 3 good + 3 defect
    assert sum(ds.labels) == 3

    sample = ds[0]
    assert sample["image"].shape == (3, 64, 64)
    assert sample["mask"].shape == (1, 64, 64)
    assert set(torch.unique(sample["mask"]).tolist()).issubset({0.0, 1.0})


def test_defect_image_has_nonempty_mask(synthetic_root):
    ds = MVTecDataset(synthetic_root, "bottle", "test", resize=64, crop=64)
    defect_idx = next(i for i, lbl in enumerate(ds.labels) if lbl == 1)
    sample = ds[defect_idx]
    assert sample["label"] == 1
    assert sample["mask"].sum() > 0


def test_unknown_category_raises(synthetic_root):
    import pytest

    with pytest.raises(ValueError):
        MVTecDataset(synthetic_root, "not_a_category", "train")
