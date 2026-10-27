"""Frozen ImageNet backbone with multi-layer feature hooks.

Both PatchCore and PaDiM treat a pretrained CNN as a *fixed* feature encoder —
no gradients, no fine-tuning. We register forward hooks on the requested ResNet
blocks (``layer1``..``layer3``) and return their activations.

Supported backbones: ``resnet18``, ``resnet50``, ``wide_resnet50_2``.
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
from torchvision import models

_BACKBONES = {
    "resnet18": (models.resnet18, "ResNet18_Weights"),
    "resnet50": (models.resnet50, "ResNet50_Weights"),
    "wide_resnet50_2": (models.wide_resnet50_2, "Wide_ResNet50_2_Weights"),
}


class FeatureExtractor(nn.Module):
    """Extract intermediate feature maps from a frozen ResNet-family backbone.

    Args:
        backbone: One of ``resnet18``, ``resnet50``, ``wide_resnet50_2``.
        layers: Block names to tap, e.g. ``["layer2", "layer3"]``.
        pretrained: Load ImageNet weights (always True in practice; exposed for
            fast unit tests that want random weights).
    """

    def __init__(
        self,
        backbone: str = "wide_resnet50_2",
        layers: List[str] | None = None,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        if backbone not in _BACKBONES:
            raise ValueError(
                f"Unsupported backbone '{backbone}'. Choose from {list(_BACKBONES)}."
            )
        self.backbone_name = backbone
        self.layers = layers or ["layer2", "layer3"]

        ctor, weights_enum_name = _BACKBONES[backbone]
        weights = None
        if pretrained:
            weights = getattr(models, weights_enum_name).IMAGENET1K_V1
        self.backbone = ctor(weights=weights)

        # Freeze: eval mode + no grad. This is what makes the methods
        # "training-free" — the backbone is never updated.
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad_(False)

        self._features: Dict[str, torch.Tensor] = {}
        self._register_hooks()

    def _register_hooks(self) -> None:
        for name in self.layers:
            module = getattr(self.backbone, name, None)
            if module is None:
                raise ValueError(f"Backbone has no layer '{name}'.")
            module.register_forward_hook(self._make_hook(name))

    def _make_hook(self, name: str):
        def hook(_module, _inp, output) -> None:
            self._features[name] = output

        return hook

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return ``{layer_name: feature_map (B, C, H, W)}`` for the input batch."""
        self._features = {}
        self.backbone(x)
        return {name: self._features[name] for name in self.layers}

    @property
    def channels(self) -> Dict[str, int]:
        """Number of channels produced by each tapped layer."""
        dims = {
            "resnet18": {"layer1": 64, "layer2": 128, "layer3": 256, "layer4": 512},
            "resnet50": {"layer1": 256, "layer2": 512, "layer3": 1024, "layer4": 2048},
            "wide_resnet50_2": {
                "layer1": 256,
                "layer2": 512,
                "layer3": 1024,
                "layer4": 2048,
            },
        }[self.backbone_name]
        return {name: dims[name] for name in self.layers}
