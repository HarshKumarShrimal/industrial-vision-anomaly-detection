"""Generate the heatmap gallery used in the README.

Fits a method on one category and renders the 4-panel
(input | GT | heatmap | overlay) figure for a few defect images.

Example::

    python scripts/generate_figures.py --config configs/patchcore.yaml \
        --category bottle --num-images 4 --out results/figures/bottle_patchcore.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import torch
from torch.utils.data import DataLoader

from iad.datasets.mvtec import MVTecDataset
from iad.runner import build_model
from iad.utils.config import load_config
from iad.utils.seed import set_seed
from iad.visualize import save_comparison_grid


def main() -> None:
    parser = argparse.ArgumentParser(description="Render anomaly heatmap grids.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--category", default="bottle")
    parser.add_argument("--root", help="Override dataset root.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-images", type=int, default=4)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.root:
        config.setdefault("data", {})["root"] = args.root
    set_seed(config.get("seed", 42))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    data = config.get("data", {})
    root = data["root"]
    resize, crop = data.get("resize", 256), data.get("crop", 224)

    train_ds = MVTecDataset(root, args.category, "train", resize, crop)
    test_ds = MVTecDataset(root, args.category, "test", resize, crop)

    model = build_model(config, device)
    model.fit(DataLoader(train_ds, batch_size=data.get("batch_size", 8)))

    # Pick defect images (label==1) for the gallery.
    defect_indices = [i for i, lbl in enumerate(test_ds.labels) if lbl == 1]
    defect_indices = defect_indices[: args.num_images]

    images, maps, masks, titles = [], [], [], []
    for idx in defect_indices:
        sample = test_ds[idx]
        pred = model.predict(sample["image"].unsqueeze(0))
        images.append(sample["image"])
        maps.append(pred.anomaly_maps[0].numpy())
        masks.append(sample["mask"])
        titles.append(f"{args.category}/{sample['defect_type']}")

    out = args.out or f"results/figures/{args.category}_{config['method']}.png"
    path = save_comparison_grid(images, maps, masks, out, titles=titles)
    print(f"Saved figure to {Path(path).resolve()}")


if __name__ == "__main__":
    main()
