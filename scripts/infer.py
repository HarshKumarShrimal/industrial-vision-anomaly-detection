#!/usr/bin/env python3
"""Command-line interface for visual anomaly inference on single images or directories."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add src to pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from iad.inference import InferencePipeline


def main():
    parser = argparse.ArgumentParser(description="Run visual anomaly detection inference on an image.")
    parser.add_argument("--image", type=str, required=True, help="Path to input image file or directory.")
    parser.add_argument("--method", type=str, default="patchcore", choices=["patchcore", "padim", "autoencoder"], help="Detection model architecture.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Anomaly decision threshold.")
    parser.add_argument("--device", type=str, default=None, help="Compute device ('cuda', 'mps', 'cpu', or 'auto').")
    parser.add_argument("--output-dir", type=str, default="results/inference", help="Directory to save output heatmaps.")
    parser.add_argument("--json-out", action="store_true", help="Print structured JSON output.")
    args = parser.parse_args()

    pipeline = InferencePipeline(
        method=args.method,
        threshold=args.threshold,
        device=args.device,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.image)
    if input_path.is_file():
        image_files = [input_path]
    elif input_path.is_dir():
        image_files = [
            f for f in input_path.iterdir()
            if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
        ]
        if not image_files:
            print(f"No image files found in {input_path}", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"Error: {input_path} is neither a file nor a directory.", file=sys.stderr)
        sys.exit(1)

    all_results = []
    for img_p in image_files:
        try:
            res = pipeline.predict(img_p, threshold=args.threshold)
            save_path = out_dir / f"{img_p.stem}_heatmap.png"
            res["heatmap_overlay"].save(save_path)
            res_summary = {
                "file": str(img_p),
                "method": args.method,
                "score": round(res["anomaly_score"], 4),
                "threshold": round(res["threshold"], 4),
                "decision": res["decision"],
                "latency_ms": res["latency_ms"],
                "heatmap_saved_to": str(save_path),
            }
            all_results.append(res_summary)
            if not args.json_out:
                print(f"[{res['decision']}] {img_p.name} | Score: {res['anomaly_score']:.4f} (Thr: {res['threshold']:.4f}) | Latency: {res['latency_ms']:.1f}ms -> Saved {save_path.name}")
        except Exception as e:
            print(f"Failed processing {img_p}: {e}", file=sys.stderr)

    if args.json_out:
        print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
