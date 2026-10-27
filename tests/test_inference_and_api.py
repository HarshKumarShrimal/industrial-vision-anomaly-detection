"""Unit and integration tests for inference pipeline, API, and device resolution."""
from __future__ import annotations

import io
import pytest
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from iad.inference import InferencePipeline
from iad.api import app
from iad.utils.device import resolve_device, set_seed


def test_device_resolution():
    cpu_dev = resolve_device("cpu")
    assert str(cpu_dev) == "cpu"

    auto_dev = resolve_device("auto")
    assert str(auto_dev) in ("cpu", "mps", "cuda")

    # Invalid string should safely fallback to cpu
    fallback_dev = resolve_device("non_existent_accelerator_xyz")
    assert str(fallback_dev) == "cpu"


def test_deterministic_seeding():
    set_seed(1234)
    r1 = np.random.rand(5)
    set_seed(1234)
    r2 = np.random.rand(5)
    np.testing.assert_allclose(r1, r2)


def test_inference_pipeline_predict_and_heatmap():
    pipe = InferencePipeline(method="patchcore", image_size=64, threshold=1.0)
    
    # Create test image
    img = np.full((64, 64, 3), 120, dtype=np.uint8)
    img[20:40, 20:40] = 255
    pil_img = Image.fromarray(img)

    result = pipe.predict(pil_img, threshold=1.0)
    assert "anomaly_score" in result
    assert "is_anomalous" in result
    assert "decision" in result
    assert "heatmap_base64" in result
    assert result["heatmap_base64"].startswith("data:image/png;base64,")
    assert isinstance(result["heatmap_overlay"], Image.Image)


def test_inference_input_validation():
    pipe = InferencePipeline(method="patchcore", image_size=64)

    # Empty bytes
    with pytest.raises(ValueError, match="empty"):
        pipe.validate_and_load_image(b"")

    # Invalid image shape
    with pytest.raises(ValueError, match="Invalid numpy image shape"):
        pipe.validate_and_load_image(np.zeros((64, 64, 7)))

    # Too small image
    with pytest.raises(ValueError, match="too small"):
        pipe.validate_and_load_image(Image.new("RGB", (4, 4)))

    # Non-existent file
    with pytest.raises(FileNotFoundError):
        pipe.validate_and_load_image("does_not_exist_image_xyz.png")


def test_fastapi_endpoints():
    client = TestClient(app)

    # Health
    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    h_data = health_resp.json()
    assert h_data["status"] == "healthy"
    assert "service" in h_data

    # UI HTML
    ui_resp = client.get("/")
    assert ui_resp.status_code == 200
    assert "Industrial Quality Inspection" in ui_resp.text

    # Predict valid image
    img = Image.new("RGB", (64, 64), color=(150, 150, 150))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    pred_resp = client.post(
        "/predict",
        files={"file": ("test.png", buf.getvalue(), "image/png")},
        data={"threshold": "0.5", "method": "patchcore"},
    )
    assert pred_resp.status_code == 200
    p_data = pred_resp.json()
    assert "anomaly_score" in p_data
    assert "is_anomalous" in p_data
    assert "heatmap_base64" in p_data
    assert p_data["heatmap_base64"].startswith("data:image/png;base64,")
