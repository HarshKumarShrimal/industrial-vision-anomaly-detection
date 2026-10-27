"""Gaussian smoothing of anomaly maps.

Both PaDiM and PatchCore upsample a coarse patch-level score map to image
resolution and then smooth it. Smoothing removes the blocky upsampling
artefacts and is part of the published recipes (sigma=4 is the PatchCore
default). We implement it as a depthwise conv so it runs on GPU and stays
differentiable-free.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _gaussian_kernel1d(sigma: float, truncate: float = 4.0) -> torch.Tensor:
    radius = max(int(truncate * sigma + 0.5), 1)
    x = torch.arange(-radius, radius + 1, dtype=torch.float32)
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    return kernel


def gaussian_smooth(maps: torch.Tensor, sigma: float = 4.0) -> torch.Tensor:
    """Apply a separable 2D Gaussian blur to a batch of anomaly maps.

    Args:
        maps: Tensor of shape ``(B, H, W)`` or ``(B, 1, H, W)``.
        sigma: Standard deviation in pixels.

    Returns:
        Smoothed maps with the same shape as the input.
    """
    squeeze_back = False
    if maps.dim() == 3:
        maps = maps.unsqueeze(1)
        squeeze_back = True

    kernel = _gaussian_kernel1d(sigma).to(maps.device, maps.dtype)
    radius = kernel.numel() // 2

    kx = kernel.view(1, 1, 1, -1)
    ky = kernel.view(1, 1, -1, 1)

    maps = F.pad(maps, (radius, radius, 0, 0), mode="reflect")
    maps = F.conv2d(maps, kx)
    maps = F.pad(maps, (0, 0, radius, radius), mode="reflect")
    maps = F.conv2d(maps, ky)

    return maps.squeeze(1) if squeeze_back else maps
