"""Plot a benchmark results CSV as a per-category AUROC bar chart.

Reads the CSV written by ``run_benchmark.py`` and renders a grouped bar chart of
image-level and pixel-level AUROC per category (the MEAN row is highlighted).
Needs only pandas + matplotlib — no GPU, no model, no dataset.

Example::

    python scripts/plot_results.py --csv results/runs/patchcore_results.csv \
        --out results/figures/patchcore_results.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Bar chart of benchmark AUROC results.")
    parser.add_argument("--csv", required=True, help="Path to <method>_results.csv.")
    parser.add_argument("--out", default=None, help="Output PNG path.")
    parser.add_argument("--title", default=None, help="Optional chart title.")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    method = str(df.get("method", pd.Series(["model"])).iloc[0])
    out = Path(args.out or Path(args.csv).with_suffix("").name + "_results.png")
    if args.out is None:
        out = Path("results/figures") / out.name

    cats = df["category"].tolist()
    img = df["image_auroc"].to_numpy(dtype=float)
    pix = df["pixel_auroc"].to_numpy(dtype=float)

    x = np.arange(len(cats))
    width = 0.4

    fig, ax = plt.subplots(figsize=(max(7, 0.7 * len(cats) + 3), 4.5))
    b1 = ax.bar(x - width / 2, img, width, label="Image AUROC", color="#4C72B0")
    b2 = ax.bar(x + width / 2, pix, width, label="Pixel AUROC", color="#DD8452")

    # Highlight the MEAN column if present.
    for i, c in enumerate(cats):
        if str(c).upper() == "MEAN":
            ax.axvspan(i - 0.5, i + 0.5, color="0.92", zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=45, ha="right")
    ax.set_ylim(0.5, 1.06)  # headroom so the vertical value labels clear the title
    ax.set_ylabel("AUROC")
    ax.set_title(args.title or f"{method.upper()} on MVTec AD — detection & localization",
                 pad=12, fontsize=12)
    ax.axhline(1.0, color="0.7", lw=0.8, ls="--")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.bar_label(b1, fmt="%.3f", padding=2, fontsize=7, rotation=90)
    ax.bar_label(b2, fmt="%.3f", padding=2, fontsize=7, rotation=90)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved chart to {out}")


if __name__ == "__main__":
    main()
