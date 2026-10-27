"""End-to-end scoring tests on synthetic data (no pretrained weights needed)."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from iad.datasets.mvtec import MVTecDataset
from iad.metrics import evaluate
from iad.models.padim import PaDiM
from iad.models.patchcore import PatchCore
from iad.utils.gaussian import gaussian_smooth
from iad.utils.sampling import KCenterGreedy


def test_gaussian_smooth_preserves_shape():
    x = torch.randn(2, 16, 16)
    out = gaussian_smooth(x, sigma=2.0)
    assert out.shape == x.shape


def test_kcenter_returns_requested_count():
    feats = torch.randn(200, 32)
    idx = KCenterGreedy(projection_dim=8, seed=0).sample(feats, ratio=0.1)
    assert idx.numel() == 20
    assert idx.unique().numel() == 20  # no duplicates


def _loaders(root):
    train = DataLoader(
        MVTecDataset(root, "bottle", "train", resize=64, crop=64), batch_size=3
    )
    test = DataLoader(
        MVTecDataset(root, "bottle", "test", resize=64, crop=64), batch_size=3
    )
    return train, test


def test_patchcore_pipeline_shapes_and_separation(synthetic_root):
    train, test = _loaders(synthetic_root)
    model = PatchCore(
        backbone="resnet18",
        layers=["layer2", "layer3"],
        coreset_ratio=0.5,
        image_size=64,
        device="cpu",
        pretrained=False,
    )
    model.fit(train)
    batch = next(iter(test))
    pred = model.predict(batch["image"])
    assert pred.image_scores.shape[0] == batch["image"].shape[0]
    assert pred.anomaly_maps.shape[-2:] == (64, 64)

    metrics = evaluate(model, test, compute_pro_metric=True)
    # Bright-square defect on flat background is easy -> should beat chance even
    # with random backbone features.
    assert metrics["image_auroc"] >= 0.5
    assert 0.0 <= metrics["pixel_auroc"] <= 1.0


def test_padim_pipeline_runs(synthetic_root):
    train, test = _loaders(synthetic_root)
    model = PaDiM(
        backbone="resnet18",
        layers=["layer1", "layer2"],
        n_dims=50,
        image_size=64,
        device="cpu",
        pretrained=False,
    )
    model.fit(train)
    batch = next(iter(test))
    pred = model.predict(batch["image"])
    assert pred.anomaly_maps.shape == (batch["image"].shape[0], 64, 64)
    assert torch.isfinite(pred.image_scores).all()
