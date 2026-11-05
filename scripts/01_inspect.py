"""Step 1: look at the data before writing any segmentation code.

Checks that every image has a mask and a split, reports the intensity range,
counts nuclei, measures nucleus size on the TRAINING split only, and saves
overlay figures so you can see what the ground truth actually looks like.

The TEST split is only checked for missing files: its images and masks are
not opened, so nothing about the test answers can influence the method.

Run it from PyCharm (right-click this file > Run) or from a terminal:
    python scripts/01_inspect.py
Optional arguments:
    --data  folder containing the unzipped BBBC039 files (default: data/raw)
    --out   where to write results (default: results/01_inspect)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save figures to files; no pop-up windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.measure import regionprops_table

from nuclei.io import (
    decode_mask,
    decode_mask_first_channel,
    find_samples,
    load_image,
    read_mask_raw,
)
from nuclei.viz import save_overlay

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TINY_AREA_PX = 30          # objects smaller than this are probably annotation debris
OVERLAYS_PER_SPLIT = 2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", type=Path, default=PROJECT_ROOT / "data" / "raw")
    p.add_argument("--out", type=Path, default=PROJECT_ROOT / "results" / "01_inspect")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    samples = find_samples(args.data)
    if not samples:
        raise SystemExit(f"No .tif images found under {args.data}. Did you unzip images.zip there?")

    # ---- 1. Do the pieces fit together? -------------------------------------
    print(f"\nFound {len(samples)} images under {args.data}")
    no_mask = [s.name for s in samples if s.mask_path is None]
    unassigned = [s.name for s in samples if s.split == "unassigned"]
    print(f"  images without a mask:  {len(no_mask)}")
    print(f"  images not in any split: {len(unassigned)}")
    for name in (no_mask + unassigned)[:5]:
        print(f"    e.g. {name}")

    test = [s for s in samples if s.split == "test"]
    print(f"  test split: {len(test)} images, {sum(s.mask_path is not None for s in test)} with a mask "
          "(not opened: only steps 4 to 7 read them, after both methods are frozen)")
    samples = [s for s in samples if s.split != "test"]

    # ---- 2. Per-image statistics (training and validation) -------------------
    rows = []
    train_props = []
    for s in samples:
        img = load_image(s.image_path)
        row = {
            "name": s.name,
            "split": s.split,
            "height": img.shape[0],
            "width": img.shape[1],
            "dtype": str(img.dtype),
            "min": int(img.min()),
            "p1": float(np.percentile(img, 1)),
            "median": float(np.median(img)),
            "p99": float(np.percentile(img, 99)),
            "max": int(img.max()),
        }
        if s.mask_path is not None:
            raw = read_mask_raw(s.mask_path)
            labels = decode_mask(raw)
            labels_ch0 = decode_mask_first_channel(raw)
            if labels.shape != img.shape:
                print(f"WARNING: shape mismatch for {s.name}: image {img.shape}, mask {labels.shape}")
            props = pd.DataFrame(
                regionprops_table(labels, properties=("label", "area", "equivalent_diameter_area"))
            )
            row.update(
                mask_channels=1 if raw.ndim == 2 else raw.shape[2],
                n_nuclei=int(labels.max()),
                n_nuclei_first_channel=int(labels_ch0.max()),
                median_area_px=float(props["area"].median()) if len(props) else np.nan,
                n_tiny=int((props["area"] < TINY_AREA_PX).sum()),
            )
            # Size statistics come from TRAINING masks only. These numbers will
            # set baseline parameters later (minimum object size, expected
            # diameter), and choosing them from test masks would leak the
            # answers into the method.
            if s.split == "training":
                train_props.append(props)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(args.out / "summary.csv", index=False)
    if "n_nuclei" not in df:
        raise SystemExit(f"No mask PNGs found under {args.data}. Did you unzip masks.zip there?")
    with_mask = df.dropna(subset=["n_nuclei"])

    # ---- 3. Splits and nucleus counts ---------------------------------------
    print("\nNuclei per split (test not inspected):")
    counts = with_mask.groupby("split")["n_nuclei"].agg(images="count", total="sum",
                                                        mean="mean", min="min", max="max")
    print(counts.round(1).to_string())

    # ---- 4. Intensity range --------------------------------------------------
    print("\nIntensity:")
    print(f"  dtypes: {sorted(df['dtype'].unique())}")
    print(f"  shapes: {sorted(set(zip(df['height'], df['width'])))}")
    print(f"  darkest pixel {df['min'].min()}, brightest pixel {df['max'].max()}")
    if df["max"].max() <= 4095:
        print("  -> every value fits in 12 bits: likely a 12-bit camera stored as 16-bit")
    print(f"  median of per-image 99th percentiles: {df['p99'].median():.0f}")

    # ---- 5. Does the simple decoding agree with the full-colour one? --------
    disagree = with_mask[with_mask["n_nuclei"] != with_mask["n_nuclei_first_channel"]]
    print(f"\nMask decoding: full-colour vs first-channel-only disagree on {len(disagree)} images")
    for _, r in disagree.head(5).iterrows():
        print(f"    {r['name']}: {r['n_nuclei']} vs {r['n_nuclei_first_channel']} nuclei")
    print(f"  objects smaller than {TINY_AREA_PX} px: {int(df['n_tiny'].sum())} in total")

    # ---- 6. Nucleus size on the training split ------------------------------
    if train_props:
        tp = pd.concat(train_props, ignore_index=True)
        a, d = tp["area"], tp["equivalent_diameter_area"]
        print(f"\nTraining-split nucleus size ({len(tp)} nuclei):")
        print(f"  area      median {a.median():.0f} px, 5th-95th pct {a.quantile(.05):.0f}-{a.quantile(.95):.0f} px")
        print(f"  diameter  median {d.median():.1f} px, 5th-95th pct {d.quantile(.05):.1f}-{d.quantile(.95):.1f} px")
        print("  (equivalent diameter = diameter of a circle with the same area)")

        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
        axes[0].hist(a, bins=80, color="#4a7bb7")
        axes[0].set_xlabel("Area (px)")
        axes[0].set_ylabel("Nuclei")
        axes[0].set_title("Nucleus area, training split")
        axes[1].hist(d, bins=80, color="#4a7bb7")
        axes[1].axvline(d.median(), color="k", ls="--", lw=1, label=f"median {d.median():.1f} px")
        axes[1].set_xlabel("Equivalent diameter (px)")
        axes[1].set_title("Nucleus diameter, training split")
        axes[1].legend()
        fig.savefig(args.out / "train_nucleus_size.png", dpi=120)
        plt.close(fig)

    # ---- 7. Overlay figures --------------------------------------------------
    for split in ["training", "validation"]:
        chosen = [s for s in samples if s.split == split and s.mask_path is not None][:OVERLAYS_PER_SPLIT]
        for s in chosen:
            img = load_image(s.image_path)
            labels = decode_mask(read_mask_raw(s.mask_path))
            save_overlay(img, labels, f"{split}: {s.name}", args.out / "overlays" / f"{split}_{s.name}.png")

    print(f"\nSaved summary.csv, train_nucleus_size.png and overlays/ to {args.out}")


if __name__ == "__main__":
    main()
