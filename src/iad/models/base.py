"""Common interface shared by all anomaly models.

Every model implements the same two-phase contract:

* ``fit(train_loader)``   — build the reference (memory bank, Gaussian stats, or
  trained weights) from *defect-free* images only.
* ``predict(images)``     — return per-image scores and pixel-level anomaly maps.

This uniform API is what lets ``run_benchmark.py`` swap methods with one line.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader


@dataclass
class AnomalyPrediction:
    """Output of a model on a batch of images.

    Attributes:
        image_scores: ``(B,)`` scalar anomaly score per image (higher = more anomalous).
        anomaly_maps: ``(B, H, W)`` pixel-level anomaly map at input resolution.
    """

    image_scores: torch.Tensor
    anomaly_maps: torch.Tensor


class AnomalyModel(abc.ABC):
    """Abstract base class for all anomaly detectors."""

    name: str = "base"

    @abc.abstractmethod
    def fit(self, train_loader: DataLoader) -> "AnomalyModel":
        """Build the reference from defect-free training images. Returns self."""

    @abc.abstractmethod
    def predict(self, images: torch.Tensor) -> AnomalyPrediction:
        """Score a batch of images already moved to the model's device."""

    def to(self, device: str | torch.device) -> "AnomalyModel":  # pragma: no cover
        self.device = torch.device(device)
        return self
