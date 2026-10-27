"""Shared fixtures: a tiny synthetic MVTec-style dataset on disk.

These let the whole pipeline be unit-tested in seconds with no dataset download
and no pretrained-weight download (models are built with ``pretrained=False``).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# Make src/ importable without an editable install.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _save(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


@pytest.fixture(scope="session")
def synthetic_root(tmp_path_factory) -> Path:
    """Create a 'bottle' category with good train + good/defect test images."""
    rng = np.random.default_rng(0)
    root = tmp_path_factory.mktemp("mvtec")
    cat = root / "bottle"

    # Train: 6 good images (smooth grey with mild noise).
    for i in range(6):
        img = (np.full((64, 64, 3), 128) + rng.integers(-10, 10, (64, 64, 3))).clip(0, 255)
        _save(cat / "train" / "good" / f"{i:03d}.png", img.astype(np.uint8))

    # Test good: 3 images.
    for i in range(3):
        img = (np.full((64, 64, 3), 128) + rng.integers(-10, 10, (64, 64, 3))).clip(0, 255)
        _save(cat / "test" / "good" / f"{i:03d}.png", img.astype(np.uint8))

    # Test defect: 3 images with a bright square, plus matching masks.
    for i in range(3):
        img = (np.full((64, 64, 3), 128) + rng.integers(-10, 10, (64, 64, 3))).clip(0, 255)
        mask = np.zeros((64, 64), dtype=np.uint8)
        img[20:40, 20:40, :] = 255
        mask[20:40, 20:40] = 255
        _save(cat / "test" / "broken" / f"{i:03d}.png", img.astype(np.uint8))
        _save(cat / "ground_truth" / "broken" / f"{i:03d}_mask.png", mask)

    return root
