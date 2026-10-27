#!/usr/bin/env python3
"""CPU-friendly smoke test script comparing PatchCore, PaDiM, and Autoencoder on synthetic data."""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from iad.runner import run_experiment
from iad.utils.config import Config


def generate_synthetic_data(base_dir: Path) -> Path:
    """Generate a tiny synthetic MVTec-like dataset structure."""
    rng = np.random.default_rng(42)
    cat_dir = base_dir / "bottle"

    # Train good: 6 images
    for i in range(6):
        img = (np.full((64, 64, 3), 128) + rng.integers(-8, 8, (64, 64, 3))).clip(0, 255).astype(np.uint8)
        p = cat_dir / "train" / "good" / f"{i:03d}.png"
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(img).save(p)

    # Test good: 3 images
    for i in range(3):
        img = (np.full((64, 64, 3), 128) + rng.integers(-8, 8, (64, 64, 3))).clip(0, 255).astype(np.uint8)
        p = cat_dir / "test" / "good" / f"{i:03d}.png"
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(img).save(p)

    # Test anomalous: 3 images
    for i in range(3):
        img = (np.full((64, 64, 3), 128) + rng.integers(-8, 8, (64, 64, 3))).clip(0, 255).astype(np.uint8)
        mask = np.zeros((64, 64), dtype=np.uint8)
        img[20:44, 20:44, :] = 255
        mask[20:44, 20:44] = 255

        img_p = cat_dir / "test" / "defect" / f"{i:03d}.png"
        mask_p = cat_dir / "ground_truth" / "defect" / f"{i:03d}_mask.png"
        img_p.parent.mkdir(parents=True, exist_ok=True)
        mask_p.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(img).save(img_p)
        Image.fromarray(mask).save(mask_p)

    return base_dir


def main():
    print("=" * 60)
    print("Industrial Vision Anomaly Detection — Smoke Benchmark")
    print("Comparing PatchCore vs PaDiM vs Autoencoder on Synthetic Data")
    print("=" * 60)

    tmp_dir = Path(tempfile.mkdtemp(prefix="iad_smoke_"))
    try:
        data_root = generate_synthetic_data(tmp_dir)
        methods = [
            ("patchcore", {
                "method": "patchcore",
                "model": {"backbone": "resnet18", "coreset_ratio": 0.2, "pretrained": False, "image_size": 64},
                "data": {"root": str(data_root), "image_size": 64, "batch_size": 2},
            }),
            ("padim", {
                "method": "padim",
                "model": {"backbone": "resnet18", "n_dims": 32, "pretrained": False, "image_size": 64},
                "data": {"root": str(data_root), "image_size": 64, "batch_size": 2},
            }),
            ("autoencoder", {
                "method": "autoencoder",
                "model": {"latent": 32, "epochs": 5, "lr": 1e-3, "image_size": 64},
                "data": {"root": str(data_root), "image_size": 64, "batch_size": 2},
            }),
        ]

        results = []
        for name, cfg_dict in methods:
            cfg = Config(cfg_dict)
            print(f"\nEvaluating {name.upper()}...")
            t0 = time.perf_counter()
            metrics = run_experiment(cfg, category="bottle", device="cpu")
            elapsed = time.perf_counter() - t0
            metrics["method"] = name
            metrics["total_elapsed_sec"] = round(elapsed, 3)
            results.append(metrics)
            print(f"  -> Image AUROC: {metrics['image_auroc']:.3f} | Pixel AUROC: {metrics['pixel_auroc']:.3f} | PRO: {metrics['pro']:.3f} | Fit Time: {metrics['fit_seconds']:.2f}s")

        df = pd.DataFrame(results)
        cols = ["method", "image_auroc", "pixel_auroc", "pro", "fit_seconds", "eval_seconds", "total_elapsed_sec"]
        df_clean = df[[c for c in cols if c in df.columns]]

        out_dir = Path("results/runs")
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "smoke_benchmark_results.csv"
        json_path = out_dir / "smoke_benchmark_results.json"
        df_clean.to_csv(csv_path, index=False)
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)

        print("\n" + "=" * 60)
        print("SUMMARY TABLE:")
        print(df_clean.to_string(index=False))
        print("=" * 60)
        print(f"Results saved to {csv_path} and {json_path}")
        print("Smoke test successfully completed!")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
