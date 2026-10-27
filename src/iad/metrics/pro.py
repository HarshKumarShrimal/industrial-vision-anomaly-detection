"""PRO — Per-Region Overlap score.

PRO is the localization metric MVTec recommends because pixel-AUROC is biased
towards large defects (a few huge regions dominate the pixel count). PRO instead
gives *every connected defect region equal weight*, regardless of size.

Definition (Bergmann et al.):
  * Sweep an anomaly-map threshold.
  * At each threshold, for every connected ground-truth region compute the
    fraction of that region that is correctly flagged; average these fractions
    -> PRO at this threshold.
  * Also compute the false-positive rate over defect-free pixels.
  * Plot PRO vs FPR and integrate up to a maximum FPR (default 0.3), then
    normalise by that limit. Higher is better; 1.0 is perfect.
"""
from __future__ import annotations

import numpy as np
from skimage.measure import label as cc_label
try:
    from scipy.integrate import trapezoid as _trapezoid
except ImportError:
    try:
        from numpy import trapezoid as _trapezoid
    except ImportError:
        from numpy import trapz as _trapezoid


def compute_pro(
    maps: np.ndarray,
    masks: np.ndarray,
    num_thresholds: int = 200,
    fpr_limit: float = 0.3,
) -> float:
    """Compute the normalised area under the PRO curve up to ``fpr_limit``."""
    maps = np.asarray(maps, dtype=np.float32)
    masks = (np.asarray(masks) > 0.5).astype(np.uint8)

    # Pre-label connected components once per image.
    regions = []  # list of (image_index, region_mask)
    for i in range(masks.shape[0]):
        if masks[i].sum() == 0:
            continue
        labelled = cc_label(masks[i])
        for region_id in range(1, labelled.max() + 1):
            regions.append((i, labelled == region_id))

    if not regions:
        return float("nan")

    inverse_masks = 1 - masks  # defect-free pixels, for FPR
    n_normal_pixels = inverse_masks.sum()

    lo, hi = float(maps.min()), float(maps.max())
    thresholds = np.linspace(lo, hi, num_thresholds)

    pros, fprs = [], []
    for thr in thresholds:
        binary = maps >= thr

        # Mean per-region true-positive overlap.
        overlaps = [
            (binary[i] & region).sum() / region.sum() for i, region in regions
        ]
        pros.append(float(np.mean(overlaps)))

        # False positive rate on defect-free pixels.
        fp = (binary & inverse_masks.astype(bool)).sum()
        fprs.append(fp / n_normal_pixels if n_normal_pixels > 0 else 0.0)

    fprs = np.asarray(fprs)
    pros = np.asarray(pros)

    # Keep the portion of the curve below the FPR limit and integrate.
    # Group duplicate FPRs and take maximum PRO
    unique_f = []
    max_p = []
    for f in np.unique(fprs):
        unique_f.append(f)
        max_p.append(float(np.max(pros[fprs == f])))
    fprs = np.asarray(unique_f, dtype=np.float64)
    pros = np.asarray(max_p, dtype=np.float64)

    # Ensure anchor at FPR=0
    if fprs[0] > 0:
        fprs = np.insert(fprs, 0, 0.0)
        pros = np.insert(pros, 0, 0.0)

    # Interpolate along a dense FPR grid from 0 to fpr_limit
    grid_fpr = np.linspace(0.0, fpr_limit, num=500)
    grid_pro = np.interp(grid_fpr, fprs, pros)

    pro_auc = float(_trapezoid(grid_pro, grid_fpr) / fpr_limit)
    return float(np.clip(pro_auc, 0.0, 1.0))
