# Interview Notes: Industrial Vision Anomaly Detection

This document provides 15 technical interview questions and comprehensive, code-referenced answers covering the architectural, mathematical, and implementation details of this project.

---

### Q1: Why is industrial anomaly detection framed as an unsupervised or "one-class" problem rather than standard supervised classification?
**Answer:**
In industrial manufacturing, defective parts are rare (often < 0.1% of production line output) and defects are infinitely varied (scratches, contamination, structural cracks, bent pins, discoloration). Training a traditional multi-class supervised classifier is impractical because one cannot collect representative training distributions of every possible future defect. Instead, models are trained strictly on *defect-free* ("good") products to learn the nominal manifold. Any observation deviating sufficiently from nominal representation is flagged as anomalous.

*Source references:* [`src/iad/models/base.py`](src/iad/models/base.py), [`src/iad/datasets/mvtec.py`](src/iad/datasets/mvtec.py).

---

### Q2: How does PatchCore work, and why does it outperform classical autoencoders?
**Answer:**
PatchCore extracts mid-level feature representations (from frozen ResNet layers `layer2` and `layer3`) across local patches of nominal images. It applies a $3 \times 3$ local neighborhood average pooling to embed spatial context into each patch token. All nominal patch tokens across the training set are collected into a memory bank. At inference time, the anomaly score of any test patch is its Euclidean distance to its nearest neighbor in the memory bank. 
Autoencoders suffer from two failure modes: (1) blurry reconstructions due to $L_2$ pixel loss, and (2) generalizing "too well" by reconstructing novel defects accurately. PatchCore relies on rich discriminative features from ImageNet backbones without needing reconstruction training.

*Source references:* [`src/iad/models/patchcore.py`](src/iad/models/patchcore.py#L40-L125).

---

### Q3: What is Coreset Subsampling in PatchCore, and why is Greedy k-Center used instead of random subsampling?
**Answer:**
A training set can generate hundreds of thousands of patch tokens, creating high memory and query latency overhead during nearest-neighbor search. Random subsampling tends to sample densely populated clusters, leaving low-density boundary modes unrepresented. PatchCore uses **Greedy k-Center subsampling** (`KCenterGreedy`), which iteratively selects the point that maximizes the minimum distance to already selected points (the minimax facility location problem). To make this computationally tractable in high dimensions, patches are first projected to a lower-dimensional space (e.g. 128-D) via a random Johnson-Lindenstrauss orthogonal matrix before greedy selection.

*Source references:* [`src/iad/utils/sampling.py`](src/iad/utils/sampling.py).

---

### Q4: How does PaDiM model nominal distributions, and how does its computational complexity compare to PatchCore?
**Answer:**
PaDiM (Patch Distribution Modeling) embeds patches at each spatial grid location $(i, j)$ across nominal training images. For each location $(i, j)$, it estimates a multivariate Gaussian distribution defined by mean vector $\mu_{i,j}$ and covariance matrix $\Sigma_{i,j}$. At test time, the anomaly score at $(i, j)$ is the Mahalanobis distance between test patch $x_{i,j}$ and $\mathcal{N}(\mu_{i,j}, \Sigma_{i,j})$.
PaDiM requires storing only position-specific parameters $(\mu, \Sigma^{-1})$, making test-time inference $O(1)$ per patch with respect to training dataset size, whereas PatchCore scales with the size of the coreset memory bank.

*Source references:* [`src/iad/models/padim.py`](src/iad/models/padim.py).

---

### Q5: What bug was discovered in the upstream PRO (Per-Region Overlap) implementation, and how was it resolved?
**Answer:**
In upstream `src/iad/metrics/pro.py`, `compute_pro` computed per-threshold overlaps and False Positive Rates (FPR). When a model localized defects perfectly, at all thresholds above zero the model flagged zero normal pixels (FPR = 0.0) while perfectly covering the defect (PRO = 1.0). When `scikit-learn.metrics.auc` was called on the filtered points with `fpr <= fpr_limit`, all x-coordinates were identical (`0.0`), collapsing $\Delta x = 0$ and producing a PRO score of `0.0`.
We resolved this by:
1. Grouping duplicate FPRs and retaining the maximum PRO.
2. Enforcing boundary anchor $(0, 0)$.
3. Resampling along a uniform, dense FPR grid on $[0, \text{fpr\_limit}]$ via piecewise linear interpolation.
4. Integrating using `scipy.integrate.trapezoid` normalized by `fpr_limit`.

*Source references:* [`src/iad/metrics/pro.py`](src/iad/metrics/pro.py#L65-L95), [`UPSTREAM.md`](UPSTREAM.md).

---

### Q6: Why is Per-Region Overlap (PRO) preferred over Pixel AUROC on MVTec AD?
**Answer:**
Pixel AUROC evaluates every pixel independently. If an image has one giant defect covering 10,000 pixels and another minor scratch covering 20 pixels, correctly segmenting the large defect while missing the scratch yields a misleadingly high Pixel AUROC (>99%). PRO treats each connected ground-truth defect component as an equal unit: for each region, it calculates the fraction of covered pixels, and averages across all regions. It evaluates this mean overlap up to an operational FPR limit (typically 30%), preventing models with excessive false alarms from scoring well.

*Source references:* [`src/iad/metrics/pro.py`](src/iad/metrics/pro.py).

---

### Q7: How is the image-level anomaly score derived from pixel-level patch scores in PatchCore?
**Answer:**
The raw image anomaly score is initialized as $s^* = \max_{m \in \mathcal{M}_{test}} s(m)$, where $s(m)$ is the nearest neighbor distance to the coreset bank. However, if $s^*$ is located in a sparse region of the memory bank, it could be an outlier. PatchCore reweights $s^*$ using the softmin ratio of distances to the $b$ nearest coreset neighbors:
$$s = \left( 1 - \frac{\exp(\|m^* - c_1\|_2)}{\sum_{j=1}^b \exp(\|m^* - c_j\|_2)} \right) \cdot s^*$$
This down-weights isolated false positives in low-density feature space.

*Source references:* [`src/iad/models/patchcore.py`](src/iad/models/patchcore.py#L113-L135).

---

### Q8: How should decision thresholds be calibrated in production deployment?
**Answer:**
The raw anomaly score is unbounded Euclidean distance or Mahalanobis distance. It cannot be thresholded at an arbitrary constant like 0.5 without calibration. In production:
1. A held-out validation set containing nominal and defect samples is passed through the model.
2. A ROC curve and Precision-Recall curve are generated.
3. Depending on factory cost tolerances, the threshold is selected to guarantee an upper bound on false positive rate (e.g. $\text{FPR} \le 1\%$) or to maximize the $F_1$ / $F_2$ score (penalizing false negatives more heavily to prevent shipping defective components).

*Source references:* [`src/iad/inference.py`](src/iad/inference.py#L125-L150).

---

### Q9: How are anomaly heatmaps generated and aligned with input image dimensions?
**Answer:**
Feature extractors downsample images (e.g. ResNet stride 8 or 16). The resulting patch anomaly map has dimensions $(H/8, W/8)$. To display an interpretable defect overlay:
1. The 2D anomaly map is bilinearly upsampled to the original image dimensions $(H, W)$.
2. A 2D Gaussian blur kernel ($\sigma = 4.0$) is convolved across the map to eliminate patch blockiness.
3. The smoothed map is min-max normalized to $[0, 1]$ and mapped through a Matplotlib Jet colormap into an RGBA heatmap.
4. The heatmap is alpha-blended ($\alpha = 0.5$) over the original RGB image.

*Source references:* [`src/iad/utils/gaussian.py`](src/iad/utils/gaussian.py), [`src/iad/inference.py`](src/iad/inference.py#L90-L115).

---

### Q10: How does your implementation support cross-platform hardware acceleration?
**Answer:**
Many PyTorch repositories hardcode `device = "cuda" if torch.cuda.is_available() else "cpu"`. This forces Apple Silicon Macs (M1/M2/M3/M4) onto single-threaded CPU. We created `src/iad/utils/device.py` with `resolve_device()`, which checks for CUDA, then checks `torch.backends.mps.is_available()` to leverage Apple Metal Performance Shaders, and falls back to CPU if no GPU accelerator is present.

*Source references:* [`src/iad/utils/device.py`](src/iad/utils/device.py).

---

### Q11: What are the primary failure modes of PatchCore in practical inspection?
**Answer:**
1. **Misalignment & Perspective Shifts:** If camera angles or product positions vary drastically, patch positions may not correspond to similar physical locations.
2. **High-Frequency Natural Textures:** Non-rigid textures (like woven textiles or carpets) produce high variance in patch embeddings, requiring larger coresets to avoid false alarms.
3. **Memory Footprint:** For 15+ complex categories, storing coresets for all backbones can demand significant RAM/VRAM during high-speed production line inference.

---

### Q12: How is the Autoencoder baseline implemented and evaluated in this project?
**Answer:**
The autoencoder consists of a 4-layer convolutional encoder compressing $224 \times 224 \times 3$ images into a compact latent bottleneck (e.g. 100-D), followed by a 4-layer transposed convolution decoder trained with Mean Squared Error (MSE) loss on nominal images. The pixel-level anomaly map is the squared reconstruction error $|x - \hat{x}|^2$ averaged across RGB channels.

*Source references:* [`src/iad/models/autoencoder.py`](src/iad/models/autoencoder.py).

---

### Q13: Why is the MVTec AD dataset restricted from being committed to GitHub?
**Answer:**
MVTec AD is published under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0). Committing commercial copies or redistributing raw images without license compliance violates their licensing terms. Our repository provides automated download scripts (`scripts/download_data.py`), comprehensive synthetic data fixtures for testing (`scripts/smoke_test.py`), and strictly ignores raw data in `.gitignore`.

*Source references:* [`.gitignore`](.gitignore), [`UPSTREAM.md`](UPSTREAM.md).

---

### Q14: How does the FastAPI service structure input validation and inference latency tracking?
**Answer:**
In `src/iad/api.py`, incoming file uploads are validated for file size, valid image encoding, and RGB channels. The `InferencePipeline` records wall-clock time with `time.perf_counter()` to return `latency_ms` in the response payload. The heatmap is rendered to a base64 Data URI PNG so web clients can render it immediately without multi-step file polling.

*Source references:* [`src/iad/api.py`](src/iad/api.py), [`src/iad/inference.py`](src/iad/inference.py).

---

### Q15: If you had to scale this inspection system to a factory conveyor belt running at 30 items per second, what would you optimize?
**Answer:**
1. **Model Quantization & ONNX/TensorRT Export:** Export the ResNet feature extractor to TensorRT or ONNX Runtime with INT8 quantization.
2. **FAISS GPU / ScaNN Indexing:** Replace standard PyTorch Euclidean distance with an inverted-file FAISS index (`faiss-gpu`) or product quantization for sub-millisecond nearest neighbor lookup.
3. **Async Batch Inference:** Batch incoming camera frames across consecutive conveyor triggers rather than running individual forward passes.
4. **Edge Deployment:** Deploy onto an edge gateway (such as NVIDIA Jetson Orin) directly interfacing with industrial GigE Vision cameras via GenICam protocol.
