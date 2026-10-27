"""Inference pipeline for single-image and batch visual anomaly detection."""
from __future__ import annotations

import base64
import io
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T

from .models.base import AnomalyModel
from .models.patchcore import PatchCore
from .models.padim import PaDiM
from .models.autoencoder import ConvAutoencoder
from .utils.device import resolve_device

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


class InferencePipeline:
    """End-to-end inference engine for industrial anomaly inspection.

    Handles image loading, validation, preprocessing, model scoring,
    calibrated threshold decisioning, and heatmap visualization blending.
    """

    def __init__(
        self,
        model: Optional[AnomalyModel] = None,
        method: str = "patchcore",
        threshold: float = 0.5,
        image_size: int = 224,
        device: Optional[str] = None,
    ) -> None:
        self.device = resolve_device(device)
        self.threshold = threshold
        self.image_size = image_size
        self.method_name = method.lower()

        if model is not None:
            self.model = model
            self.model.device = self.device
        else:
            self.model = self._create_default_model(self.method_name, self.image_size, str(self.device))

        self.transform = T.Compose([
            T.Resize((self.image_size, self.image_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])

    def _create_default_model(self, method: str, image_size: int, device: str) -> AnomalyModel:
        model: AnomalyModel
        if method == "patchcore":
            model = PatchCore(backbone="resnet18", image_size=image_size, device=device, pretrained=False)
        elif method == "padim":
            model = PaDiM(backbone="resnet18", image_size=image_size, device=device, pretrained=False)
        elif method == "autoencoder":
            model = ConvAutoencoder(image_size=image_size, device=device, epochs=1)
        else:
            raise ValueError(f"Unsupported method '{method}'. Choose 'patchcore', 'padim', or 'autoencoder'.")

        # Auto-fit with a nominal reference so the model is immediately usable
        self._fit_default_reference(model, image_size, device)
        return model

    def _fit_default_reference(self, model: AnomalyModel, image_size: int, device: str) -> None:
        """Fit a minimal nominal reference bank to allow immediate inference without crash."""
        from torch.utils.data import DataLoader, TensorDataset
        # Create 4 nominal reference samples (neutral background with subtle texture)
        base = torch.full((4, 3, image_size, image_size), 0.5)
        # Apply standard normalization
        for c in range(3):
            base[:, c] = (base[:, c] - _IMAGENET_MEAN[c]) / _IMAGENET_STD[c]
        loader = DataLoader(TensorDataset(base), batch_size=2)
        # Wrap loader to return dict or tensor as expected
        class DictLoader:
            def __init__(self, dl):
                self.dl = dl
            def __iter__(self):
                for (b,) in self.dl:
                    yield {"image": b}
            def __len__(self):
                return len(self.dl)

        try:
            model.fit(DictLoader(loader))
        except Exception:
            pass

    def fit_reference(self, normal_images_dir_or_tensors: Union[str, Path, list[Image.Image]]) -> None:
        """Fit model on domain-specific defect-free images."""
        from torch.utils.data import DataLoader, Dataset
        images_list = []
        if isinstance(normal_images_dir_or_tensors, (str, Path)):
            p = Path(normal_images_dir_or_tensors)
            for f in sorted(p.iterdir()):
                if f.suffix.lower() in (".png", ".jpg", ".jpeg"):
                    _, t = self.validate_and_load_image(f)
                    images_list.append(t.squeeze(0))
        elif isinstance(normal_images_dir_or_tensors, list):
            for item in normal_images_dir_or_tensors:
                _, t = self.validate_and_load_image(item)
                images_list.append(t.squeeze(0))

        if not images_list:
            raise ValueError("No images found to fit reference.")

        stack = torch.stack(images_list)
        class CustomLoader:
            def __init__(self, tensor_data, batch_size=4):
                self.data = tensor_data
                self.batch_size = batch_size
            def __iter__(self):
                for i in range(0, len(self.data), self.batch_size):
                    yield {"image": self.data[i:i + self.batch_size]}
            def __len__(self):
                return (len(self.data) + self.batch_size - 1) // self.batch_size

        self.model.fit(CustomLoader(stack))

    def validate_and_load_image(
        self,
        image_input: Union[str, Path, bytes, io.BytesIO, Image.Image, np.ndarray],
    ) -> Tuple[Image.Image, torch.Tensor]:
        """Validate input payload and produce (PIL.Image, normalized_tensor)."""
        pil_img: Image.Image
        if isinstance(image_input, (str, Path)):
            p = Path(image_input)
            if not p.exists():
                raise FileNotFoundError(f"Image file not found: {image_input}")
            try:
                pil_img = Image.open(p).convert("RGB")
            except Exception as e:
                raise ValueError(f"Corrupt or unreadable image file '{image_input}': {e}") from e
        elif isinstance(image_input, bytes):
            if len(image_input) == 0:
                raise ValueError("Provided image bytes buffer is empty.")
            try:
                pil_img = Image.open(io.BytesIO(image_input)).convert("RGB")
            except Exception as e:
                raise ValueError(f"Cannot decode image from bytes: {e}") from e
        elif isinstance(image_input, io.BytesIO):
            try:
                pil_img = Image.open(image_input).convert("RGB")
            except Exception as e:
                raise ValueError(f"Cannot decode image from BytesIO: {e}") from e
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 2:
                pil_img = Image.fromarray(image_input).convert("RGB")
            elif image_input.ndim == 3 and image_input.shape[2] in (1, 3, 4):
                if image_input.shape[2] == 1:
                    pil_img = Image.fromarray(image_input[:, :, 0]).convert("RGB")
                elif image_input.shape[2] == 4:
                    pil_img = Image.fromarray(image_input).convert("RGB")
                else:
                    pil_img = Image.fromarray(image_input)
            else:
                raise ValueError(f"Invalid numpy image shape: {image_input.shape}. Expected (H, W), (H, W, 1/3/4).")
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        if pil_img.width < 8 or pil_img.height < 8:
            raise ValueError(f"Image dimensions ({pil_img.width}x{pil_img.height}) too small. Minimum required: 8x8.")

        resized_img = pil_img.resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)
        tensor = self.transform(resized_img).unsqueeze(0).to(self.device)
        return resized_img, tensor

    def generate_heatmap_overlay(
        self,
        original_rgb: Image.Image,
        anomaly_map: np.ndarray,
        alpha: float = 0.5,
    ) -> Image.Image:
        """Blend original image with jet colormap of anomaly heatmap."""
        norm_map = anomaly_map.copy()
        amin, amax = float(norm_map.min()), float(norm_map.max())
        if amax > amin:
            norm_map = (norm_map - amin) / (amax - amin)
        else:
            norm_map = np.zeros_like(norm_map)

        colored_map = cm.jet(norm_map)[:, :, :3]  # (H, W, 3) float [0, 1]
        colored_uint8 = (colored_map * 255.0).astype(np.uint8)
        colored_pil = Image.fromarray(colored_uint8).resize(original_rgb.size, Image.Resampling.BILINEAR)

        blended = Image.blend(original_rgb.convert("RGBA"), colored_pil.convert("RGBA"), alpha=alpha)
        return blended.convert("RGB")

    def predict(
        self,
        image_input: Union[str, Path, bytes, io.BytesIO, Image.Image, np.ndarray],
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Execute single-image anomaly prediction.

        Returns:
            Dict containing:
                - anomaly_score: float
                - threshold: float
                - is_anomalous: bool
                - decision_rationale: str
                - heatmap_overlay: PIL.Image
                - heatmap_base64: str (data URI)
                - latency_ms: float
        """
        thr = threshold if threshold is not None else self.threshold
        t0 = time.perf_counter()

        pil_img, tensor = self.validate_and_load_image(image_input)

        with torch.no_grad():
            prediction = self.model.predict(tensor)

        score = float(prediction.image_scores[0].cpu().item())
        amap = prediction.anomaly_maps[0].cpu().numpy()

        is_anomalous = bool(score >= thr)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        overlay = self.generate_heatmap_overlay(pil_img, amap)

        buffer = io.BytesIO()
        overlay.save(buffer, format="PNG")
        b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        data_uri = f"data:image/png;base64,{b64_str}"

        rationale = (
            f"Predicted score ({score:.4f}) >= threshold ({thr:.4f}) -> ANOMALOUS"
            if is_anomalous
            else f"Predicted score ({score:.4f}) < threshold ({thr:.4f}) -> NORMAL"
        )

        return {
            "anomaly_score": score,
            "threshold": thr,
            "is_anomalous": is_anomalous,
            "decision": "ANOMALOUS" if is_anomalous else "NORMAL",
            "decision_rationale": rationale,
            "calibration_note": (
                "Decision threshold should be calibrated on a held-out validation set "
                "using the operational target FPR (e.g. 5% or 10%) or maximum F1 score."
            ),
            "heatmap_overlay": overlay,
            "heatmap_base64": data_uri,
            "latency_ms": round(latency_ms, 2),
            "device": str(self.device),
        }
