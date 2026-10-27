"""Download MVTec AD and lay it out in the standard folder structure.

MVTec rotated its official download link, so this script pulls the dataset from a
public, ungated Hugging Face mirror (``TheoM55/mvtec_all_objects_split``) that
hosts all 15 categories as parquet with ``image_path``, ``mask_path``, named
``defect`` types and a binary ``label``. It then materializes the canonical MVTec
layout on disk::

    <root>/<category>/
        train/good/*.png
        test/good/*.png
        test/<defect_type>/*.png
        ground_truth/<defect_type>/*_mask.png

so the file-based loader in ``iad/datasets/mvtec.py`` works unchanged.

Requirements: ``pip install datasets pillow`` (both are in requirements.txt).
No Hugging Face account or token is required — the mirror is public.

Examples::

    # All 15 categories (~5.3 GB)
    python scripts/download_data.py --root ./data/mvtec_ad

    # Just a couple of categories (much smaller, good for a quick run)
    python scripts/download_data.py --root ./data/mvtec_ad --categories capsule bottle

On Kaggle you can instead add the "MVTec AD" dataset and skip this entirely by
passing ``--root /kaggle/input/mvtec-ad`` to the benchmark scripts.
"""
from __future__ import annotations

import argparse
from pathlib import Path

HF_REPO = "TheoM55/mvtec_all_objects_split"

MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "hazelnut", "metal_nut",
    "pill", "screw", "toothbrush", "transistor", "zipper",
    "carpet", "grid", "leather", "tile", "wood",
]


def _save_png(image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _already_done(cat_dir: Path) -> bool:
    good = cat_dir / "train" / "good"
    return good.is_dir() and any(good.glob("*.png"))


def materialize_category(category: str, root: Path) -> None:
    """Download one category from the HF mirror and write the MVTec layout."""
    from datasets import load_dataset

    cat_dir = root / category
    if _already_done(cat_dir):
        print(f"[skip] {category} already present.")
        return

    for split in ("train", "test"):
        print(f"[download] {category}.{split} from {HF_REPO} ...")
        ds = load_dataset(HF_REPO, split=f"{category}.{split}")
        counters: dict[str, int] = {}
        for ex in ds:
            label = int(ex["label"])
            defect = ex.get("defect") or ("good" if label == 0 else "anomaly")
            stem = f"{counters.get(defect, 0):04d}"
            counters[defect] = counters.get(defect, 0) + 1

            _save_png(ex["image_path"].convert("RGB"),
                      cat_dir / split / defect / f"{stem}.png")

            if split == "test" and label != 0:
                mask = ex.get("mask_path")
                if mask is not None:
                    _save_png(mask.convert("L"),
                              cat_dir / "ground_truth" / defect / f"{stem}_mask.png")

    n_train = len(list((cat_dir / "train" / "good").glob("*.png")))
    n_test = len(list((cat_dir / "test").rglob("*.png")))
    print(f"[done] {category}: {n_train} train (good) + {n_test} test images")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download MVTec AD via a public Hugging Face mirror.")
    parser.add_argument(
        "--root",
        default="./data/mvtec_ad",
        help="Directory that will hold the 15 category sub-folders.",
    )
    parser.add_argument(
        "--categories",
        nargs="*",
        default=None,
        help="Subset of categories to download (default: all 15).",
    )
    args = parser.parse_args()

    try:
        import datasets  # noqa: F401
    except ImportError:
        raise SystemExit(
            "This downloader needs the 'datasets' package:\n"
            "    pip install datasets pillow\n"
            "(both are listed in requirements.txt)."
        )

    root = Path(args.root)
    categories = args.categories or MVTEC_CATEGORIES
    unknown = [c for c in categories if c not in MVTEC_CATEGORIES]
    if unknown:
        raise SystemExit(f"Unknown categories {unknown}. Choose from {MVTEC_CATEGORIES}.")

    print(f"Downloading {len(categories)} categories into {root.resolve()}")
    for category in categories:
        materialize_category(category, root)

    print(f"\n[all done] MVTec AD ready under: {root.resolve()}")


if __name__ == "__main__":
    main()
