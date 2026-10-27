"""Convolutional autoencoder reconstruction baseline.

Unlike PaDiM and PatchCore, this baseline *is* trained — but only on defect-free
images, and it is deliberately small so it trains in a few minutes on a free
Colab T4. It exists to answer a reviewer's natural question: "how much does a
frozen-backbone memory method actually buy you over a classic reconstruction
autoencoder?"

The model learns to reconstruct good images. At test time, regions it cannot
reconstruct well (defects it never saw) produce a large per-pixel reconstruction
error, which becomes the anomaly map.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..utils.device import resolve_device
from ..utils.gaussian import gaussian_smooth
from .base import AnomalyModel, AnomalyPrediction


class _Encoder(nn.Module):
    def __init__(self, latent: int = 100) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(True),     # 112
            nn.Conv2d(32, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True),    # 56
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True),  # 28
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(True), # 14
            nn.Conv2d(256, latent, 4, 2, 1),                                  # 7
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _Decoder(nn.Module):
    def __init__(self, latent: int = 100) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvAutoencoder(AnomalyModel):
    name = "autoencoder"

    def __init__(
        self,
        latent: int = 100,
        epochs: int = 50,
        lr: float = 2e-3,
        image_size: int = 224,
        sigma: float = 4.0,
        device: str = "cuda",
        seed: int = 42,
    ) -> None:
        self.device = resolve_device(device)
        self.encoder = _Encoder(latent).to(self.device)
        self.decoder = _Decoder(latent).to(self.device)
        self.epochs = epochs
        self.lr = lr
        self.image_size = image_size
        self.sigma = sigma
        self.seed = seed

    # ------------------------------------------------------------------ #
    def fit(self, train_loader: DataLoader) -> "ConvAutoencoder":
        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        optimizer = torch.optim.Adam(params, lr=self.lr)
        self.encoder.train()
        self.decoder.train()
        for epoch in range(self.epochs):
            running = 0.0
            for batch in tqdm(
                train_loader, desc=f"AE epoch {epoch + 1}/{self.epochs}", leave=False
            ):
                images = batch["image"].to(self.device)
                optimizer.zero_grad()
                recon = self.decoder(self.encoder(images))
                loss = F.mse_loss(recon, images)
                loss.backward()
                optimizer.step()
                running += loss.item() * images.size(0)
        self.encoder.eval()
        self.decoder.eval()
        return self

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> AnomalyPrediction:
        images = images.to(self.device)
        recon = self.decoder(self.encoder(images))
        # Per-pixel L2 error across channels.
        error = ((images - recon) ** 2).mean(dim=1)  # (B, H, W)
        error = F.interpolate(
            error.unsqueeze(1),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        score_map = gaussian_smooth(error, sigma=self.sigma)
        image_scores = score_map.flatten(1).max(dim=1).values
        return AnomalyPrediction(
            image_scores=image_scores.cpu(), anomaly_maps=score_map.cpu()
        )
