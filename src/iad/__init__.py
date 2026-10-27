"""iad — training-free industrial anomaly detection & localization on MVTec AD.

Public API re-exports the most commonly used building blocks so that
notebooks and scripts can simply do::

    from iad import PatchCore, PaDiM, MVTecDataset, evaluate
"""
from __future__ import annotations

from .datasets.mvtec import MVTEC_CATEGORIES, MVTecDataset
from .metrics import compute_image_auroc, compute_pixel_auroc, compute_pro, evaluate
from .models.autoencoder import ConvAutoencoder
from .models.padim import PaDiM
from .models.patchcore import PatchCore

__all__ = [
    "MVTEC_CATEGORIES",
    "MVTecDataset",
    "PatchCore",
    "PaDiM",
    "ConvAutoencoder",
    "compute_image_auroc",
    "compute_pixel_auroc",
    "compute_pro",
    "evaluate",
]

__version__ = "0.1.0"
