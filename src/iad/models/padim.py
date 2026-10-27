"""PaDiM — Patch Distribution Modeling.

Reference: Defard et al., "PaDiM: a Patch Distribution Modeling Framework for
Anomaly Detection and Localization" (ICPR 2021).

Idea
----
1. Extract multi-layer features from a frozen backbone and concatenate them
   (after upsampling to a common spatial grid). Each spatial position now owns
   an embedding vector that fuses semantic levels.
2. Randomly keep a subset of the embedding channels — the paper shows random
   selection works as well as PCA and is far cheaper.
3. Model every spatial position ``(i, j)`` with a multivariate Gaussian
   ``N(mu_ij, Sigma_ij)`` estimated from the defect-free training set.
4. At test time the anomaly score of a position is the Mahalanobis distance to
   its Gaussian. Far-from-normal patches light up.

No gradient training happens — only mean/covariance estimation.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..utils.device import resolve_device
from ..utils.gaussian import gaussian_smooth
from .base import AnomalyModel, AnomalyPrediction
from .feature_extractor import FeatureExtractor

# Default number of randomly-kept channels per backbone (from the paper /
# the widely-used anomalib defaults).
_DEFAULT_N_DIMS = {
    "resnet18": 100,
    "resnet50": 550,
    "wide_resnet50_2": 550,
}


class PaDiM(AnomalyModel):
    name = "padim"

    def __init__(
        self,
        backbone: str = "wide_resnet50_2",
        layers: List[str] | None = None,
        n_dims: int | None = None,
        image_size: int = 224,
        sigma: float = 4.0,
        epsilon: float = 0.01,
        device: str = "cuda",
        seed: int = 42,
        pretrained: bool = True,
    ) -> None:
        self.device = resolve_device(device)
        layers = layers or ["layer1", "layer2", "layer3"]
        self.extractor = FeatureExtractor(backbone, layers, pretrained=pretrained).to(self.device)
        self.layers = layers
        self.image_size = image_size
        self.sigma = sigma
        self.epsilon = epsilon
        self.seed = seed

        total_channels = sum(self.extractor.channels.values())
        n_dims = n_dims or _DEFAULT_N_DIMS.get(backbone, 550)
        n_dims = min(n_dims, total_channels)
        generator = torch.Generator().manual_seed(seed)
        self.dim_idx = torch.randperm(total_channels, generator=generator)[:n_dims]

        # Learned statistics, filled by .fit().
        self.mean: torch.Tensor | None = None
        self.inv_cov: torch.Tensor | None = None
        self.feat_hw: int | None = None

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _embed(self, images: torch.Tensor) -> torch.Tensor:
        """Return concatenated, dim-reduced embeddings of shape ``(B, C, H, W)``."""
        feats = self.extractor(images.to(self.device))
        ref = feats[self.layers[0]]
        target_hw = ref.shape[-2:]
        parts = [ref]
        for name in self.layers[1:]:
            parts.append(
                F.interpolate(feats[name], size=target_hw, mode="nearest")
            )
        embedding = torch.cat(parts, dim=1)
        embedding = embedding[:, self.dim_idx, :, :]
        return embedding

    # ------------------------------------------------------------------ #
    def fit(self, train_loader: DataLoader) -> "PaDiM":
        embeddings = []
        for batch in tqdm(train_loader, desc="PaDiM: embedding train set", leave=False):
            emb = self._embed(batch["image"]).cpu()
            embeddings.append(emb)
        embeddings = torch.cat(embeddings, dim=0)  # (N, C, H, W)

        n, c, h, w = embeddings.shape
        self.feat_hw = h
        # (HW, N, C)
        embeddings = embeddings.permute(2, 3, 0, 1).reshape(h * w, n, c)

        mean = embeddings.mean(dim=1)  # (HW, C)
        centered = embeddings - mean.unsqueeze(1)
        cov = torch.einsum("pnc,pnd->pcd", centered, centered) / max(n - 1, 1)
        identity = torch.eye(c).unsqueeze(0)
        cov = cov + self.epsilon * identity

        # Precompute inverse covariance once; reused for every test image.
        self.inv_cov = torch.linalg.inv(cov).to(self.device)
        self.mean = mean.to(self.device)
        return self

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> AnomalyPrediction:
        if self.mean is None or self.inv_cov is None:
            raise RuntimeError("Call .fit() before .predict().")

        emb = self._embed(images)  # (B, C, H, W)
        b, c, h, w = emb.shape
        emb = emb.permute(2, 3, 0, 1).reshape(h * w, b, c)  # (HW, B, C)

        delta = emb - self.mean.unsqueeze(1)  # (HW, B, C)
        # Mahalanobis^2 = delta^T Sigma^-1 delta, batched over positions & images.
        left = torch.einsum("pbc,pcd->pbd", delta, self.inv_cov)
        dist_sq = torch.einsum("pbd,pbd->pb", left, delta)  # (HW, B)
        dist = torch.sqrt(dist_sq.clamp_min(0.0))

        score_map = dist.reshape(h, w, b).permute(2, 0, 1)  # (B, H, W)
        score_map = F.interpolate(
            score_map.unsqueeze(1),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        score_map = gaussian_smooth(score_map, sigma=self.sigma)

        image_scores = score_map.flatten(1).max(dim=1).values
        return AnomalyPrediction(image_scores=image_scores.cpu(), anomaly_maps=score_map.cpu())
