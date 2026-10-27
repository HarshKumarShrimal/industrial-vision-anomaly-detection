# Upstream Attribution & Engineering Contributions

## Upstream Repository Details
- **Upstream Repository:** [mtaha-ai/industrial-anomaly-detection](https://github.com/mtaha-ai/industrial-anomaly-detection)
- **Original Author:** Mohamed Taha (@mtaha-ai)
- **Upstream License:** MIT License (retained in `LICENSE`)
- **Dataset Reference:** MVTec Anomaly Detection Dataset (MVTec AD) by Bergmann et al. (CC BY-NC-SA 4.0). The dataset is strictly excluded from version control in compliance with non-commercial license requirements.

---

## Retained Upstream Components
- Core architecture abstractions for unsupervised anomaly detection (`AnomalyModel`, `AnomalyPrediction`).
- Implementations of PatchCore memory-bank neighbor search and PaDiM multivariate Gaussian modeling.
- Base convolutional autoencoder reconstruction pipeline.
- MVTec AD PyTorch dataset loader and feature extraction hooks on frozen ResNet backbones.

---

## Substantive Engineering Contributions by Harsh Kumar Shrimal

1. **Fixed Critical PRO Metric Integration Bug:**
   - Discovered and resolved a fundamental bug in `src/iad/metrics/pro.py` where perfect anomaly localization scored `0.0` AUROC/PRO because duplicate false positive rates (FPR) collapsed trapezoidal width to zero.
   - Re-implemented numerically stable monotonic grouping, boundary anchor enforcement at $(0, 0)$, dense regular FPR interpolation across $[0, \text{fpr\_limit}]$, and robust trapezoidal integration using `scipy.integrate.trapezoid` / `np.trapezoid`.
   - Enabled all 14 baseline test suites to pass with 0 warnings.

2. **End-to-End Single-Image Inference Pipeline (`src/iad/inference.py`):**
   - Engineered the `InferencePipeline` supporting PIL images, raw bytes, numpy arrays, and file paths.
   - Built automatic nominal baseline reference caching to prevent runtime crashes when invoking inference without explicit training runs.
   - Integrated Jet colormap heatmap synthesis and original image alpha-blending for visual inspection.
   - Added decision rationale generation with configurable thresholding and formal calibration documentation.

3. **Production FastAPI Service & Interactive Web UI (`src/iad/api.py`, `scripts/serve.py`):**
   - Implemented a complete REST API with `/health` and `/predict` endpoints.
   - Embedded an interactive browser dashboard enabling drag-and-drop image uploads, model architecture selection, interactive threshold sliders, and side-by-side original versus defect heatmap overlays.

4. **CLI Inference Tool (`scripts/infer.py`):**
   - Built a flexible CLI supporting single-image and batch-directory processing with structured JSON or tabular terminal output.

5. **Cross-Platform Device Resolution (`src/iad/utils/device.py`):**
   - Replaced rigid CUDA-only fallbacks with intelligent multi-accelerator device resolution supporting Apple Silicon Metal Performance Shaders (`mps`), NVIDIA `cuda`, and `cpu`.

6. **Reproducible CPU Smoke Benchmark (`scripts/smoke_test.py`):**
   - Developed a standalone synthetic dataset generation harness allowing full comparative benchmarking (PatchCore vs PaDiM vs Autoencoder) in under 90 seconds without downloading external dataset archives.
   - Saved reproducible JSON and CSV benchmark records in `results/runs/`.

7. **CI Pipeline & Expanded Test Suite:**
   - Added comprehensive integration tests in `tests/test_inference_and_api.py` covering device resolution, input edge cases, image validation, and API contracts (expanding passing tests to 19).
   - Configured GitHub Actions CI workflow in `.github/workflows/ci.yml`.
