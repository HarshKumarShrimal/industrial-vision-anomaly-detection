"""FastAPI service and Web UI for Industrial Vision Anomaly Detection."""
from __future__ import annotations

import io
from typing import Optional
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from .inference import InferencePipeline

app = FastAPI(
    title="Industrial Quality Inspection — Anomaly Detection API",
    description="Deep Visual Anomaly Detection (PatchCore, PaDiM, Autoencoder) on Industrial Components.",
    version="1.0.0",
)

# Global pipeline instance (default PatchCore)
_pipeline: Optional[InferencePipeline] = None


def get_pipeline(method: str = "patchcore", threshold: float = 0.5) -> InferencePipeline:
    global _pipeline
    if _pipeline is None or _pipeline.method_name != method:
        _pipeline = InferencePipeline(method=method, threshold=threshold)
    else:
        _pipeline.threshold = threshold
    return _pipeline


class PredictResponse(BaseModel):
    anomaly_score: float
    threshold: float
    is_anomalous: bool
    decision: str
    decision_rationale: str
    calibration_note: str
    heatmap_base64: str
    latency_ms: float
    device: str


@app.get("/health")
def health() -> dict:
    pipe = get_pipeline()
    return {
        "status": "healthy",
        "service": "industrial-vision-anomaly-detection",
        "method": pipe.method_name,
        "device": str(pipe.device),
        "default_threshold": pipe.threshold,
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(...),
    threshold: float = Form(0.5),
    method: str = Form("patchcore"),
):
    try:
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        pipeline = get_pipeline(method=method, threshold=threshold)
        result = pipeline.predict(content, threshold=threshold)

        return PredictResponse(
            anomaly_score=result["anomaly_score"],
            threshold=result["threshold"],
            is_anomalous=result["is_anomalous"],
            decision=result["decision"],
            decision_rationale=result["decision_rationale"],
            calibration_note=result["calibration_note"],
            heatmap_base64=result["heatmap_base64"],
            latency_ms=result["latency_ms"],
            device=result["device"],
        )
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Industrial Visual Anomaly Detection</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 24px; }
    .container { max-width: 960px; margin: 0 auto; background: #1e293b; padding: 32px; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.5); }
    h1 { color: #38bdf8; margin-top: 0; }
    p { color: #94a3b8; line-height: 1.5; }
    .form-group { margin-bottom: 20px; }
    label { display: block; margin-bottom: 6px; font-weight: 600; color: #cbd5e1; }
    input[type="file"], select, input[type="range"] { width: 100%; padding: 10px; background: #0f172a; border: 1px solid #334155; border-radius: 6px; color: #f8fafc; box-sizing: border-box; }
    button { background: #0284c7; color: white; border: none; padding: 12px 24px; border-radius: 6px; font-size: 16px; font-weight: 600; cursor: pointer; transition: background 0.2s; }
    button:hover { background: #0369a1; }
    .results { margin-top: 32px; padding: 24px; background: #0f172a; border-radius: 8px; border: 1px solid #334155; display: none; }
    .badge { display: inline-block; padding: 6px 14px; border-radius: 9999px; font-weight: 700; font-size: 14px; text-transform: uppercase; margin-bottom: 12px; }
    .badge.normal { background: #166534; color: #86efac; }
    .badge.anomaly { background: #991b1b; color: #fca5a5; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 16px; }
    .preview-box { text-align: center; background: #1e293b; padding: 12px; border-radius: 6px; }
    .preview-box img { max-width: 100%; height: auto; border-radius: 4px; }
  </style>
</head>
<body>
  <div class="container">
    <h1>Industrial Quality Inspection</h1>
    <p>Deep visual anomaly detection & localization using <strong>PatchCore</strong>, <strong>PaDiM</strong>, and <strong>Convolutional Autoencoders</strong>.</p>
    
    <div class="form-group">
      <label for="imageInput">Upload Manufactured Component Image:</label>
      <input type="file" id="imageInput" accept="image/*">
    </div>

    <div class="form-group">
      <label for="methodSelect">Detection Architecture:</label>
      <select id="methodSelect">
        <option value="patchcore">PatchCore (Nearest Neighbor Memory Bank)</option>
        <option value="padim">PaDiM (Patch Distribution Modeling)</option>
        <option value="autoencoder">Convolutional Autoencoder Baseline</option>
      </select>
    </div>

    <div class="form-group">
      <label for="thrRange">Anomaly Decision Threshold: <span id="thrVal">0.50</span></label>
      <input type="range" id="thrRange" min="0.05" max="2.0" step="0.05" value="0.50" oninput="document.getElementById('thrVal').textContent=parseFloat(this.value).toFixed(2)">
    </div>

    <button onclick="runInference()">Run Inspection</button>

    <div class="results" id="resultsBox">
      <div id="badge"></div>
      <div id="stats" style="margin-bottom: 12px; color: #cbd5e1;"></div>
      <div class="grid">
        <div class="preview-box">
          <h4>Original Image</h4>
          <img id="origImg" src="" alt="Original">
        </div>
        <div class="preview-box">
          <h4>Defect Heatmap Overlay</h4>
          <img id="heatmapImg" src="" alt="Heatmap">
        </div>
      </div>
      <p style="margin-top: 16px; font-size: 13px; color: #64748b;" id="calibrationNote"></p>
    </div>
  </div>

  <script>
    async function runInference() {
      const fileInput = document.getElementById('imageInput');
      if (!fileInput.files[0]) { alert('Please select an image first.'); return; }
      
      const file = fileInput.files[0];
      const method = document.getElementById('methodSelect').value;
      const thr = document.getElementById('thrRange').value;

      const formData = new FormData();
      formData.append('file', file);
      formData.append('threshold', thr);
      formData.append('method', method);

      // Display original preview
      const reader = new FileReader();
      reader.onload = e => document.getElementById('origImg').src = e.target.result;
      reader.readAsDataURL(file);

      const res = await fetch('/predict', { method: 'POST', body: formData });
      if (!res.ok) {
        const err = await res.json();
        alert('Error: ' + (err.detail || 'Inference failed'));
        return;
      }
      const data = await res.json();
      
      const resBox = document.getElementById('resultsBox');
      resBox.style.display = 'block';

      const badge = document.getElementById('badge');
      badge.textContent = data.decision;
      badge.className = 'badge ' + (data.is_anomalous ? 'anomaly' : 'normal');

      document.getElementById('stats').innerHTML = `
        <strong>Anomaly Score:</strong> ${data.anomaly_score.toFixed(4)} | 
        <strong>Threshold:</strong> ${data.threshold.toFixed(4)} | 
        <strong>Latency:</strong> ${data.latency_ms} ms (${data.device})
      `;
      document.getElementById('heatmapImg').src = data.heatmap_base64;
      document.getElementById('calibrationNote').textContent = data.calibration_note;
    }
  </script>
</body>
</html>
"""
