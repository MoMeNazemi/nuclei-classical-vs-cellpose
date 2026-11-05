"""Step 7: one summary figure for the README and for posting.

Reads the frozen test results (steps 4 and 5) and draws them into
docs/figures/summary.png. Nothing is recomputed except the drawing, and
every number on the figure comes from the saved results.

Run from PyCharm (right-click > Run) or:
    python scripts/07_summary_figure.py
    python scripts/07_summary_figure.py --repo github.com/MoMeNazemi/nuclei-classical-vs-cellpose   (footer link)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from matplotlib.patches import Patch

from nuclei.evaluate import IOU_THRESHOLDS, evaluate, f1_from_counts
from nuclei.io import find_samples, load_image, load_mask
from nuclei.plots import COLOURS, INK, MUTED, NAMES, RULE
from nuclei.viz import ERROR_COLOURS, ERROR_NAMES, annotated_rgb, error_rgb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PUBLISHED_GREY = "#a9a8a1"
CROP = 240  # px, side of the image crops in panel A
W, H = 15.0, 9.8  # figure size, inches


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", type=Path, default=PROJECT_ROOT / "data" / "raw")
    p.add_argument("--results", type=Path, default=PROJECT_ROOT / "results")
    p.add_argument("--out", type=Path, default=PROJECT_ROOT / "docs" / "figures" / "summary.png")
    p.add_argument("--repo", default="", help="repository address for the footer, e.g. github.com/you/nuclei-classical-vs-cellpose")
    return p.parse_args()


def busiest_window(labels: np.ndarray, size: int) -> tuple[int, int]:
    """Top-left corner of the size x size window holding the most annotated objects."""
    ys, xs = np.nonzero(labels)
    lab = labels[ys, xs]
    n = np.bincount(lab)
    keep = n > 0
    cy = np.bincount(lab, weights=ys)[keep] / n[keep]
    cx = np.bincount(lab, weights=xs)[keep] / n[keep]
    best, corner = -1, (0, 0)
    for y0 in range(0, max(labels.shape[0] - size, 0) + 1, 10):
        for x0 in range(0, max(labels.shape[1] - size, 0) + 1, 10):
            k = int(((cy >= y0) & (cy < y0 + size) & (cx >= x0) & (cx < x0 + size)).sum())
            if k > best:
                best, corner = k, (y0, x0)
    return corner


def style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=INK, labelsize=9)
    ax.set_axisbelow(True)


def heading(fig, x: float, y: float, letter: str, title: str) -> None:
    fig.text(x, y, letter, fontsize=12, fontweight="bold", color=INK)
    fig.text(x + 0.018, y, title, fontsize=11, color=INK)


def main() -> None:
    args = parse_args()
    test = json.loads((args.results / "04_test" / "summary.json").read_text())
    per_image = pd.read_csv(args.results / "04_test" / "per_image.csv")
    objects = pd.read_csv(args.results / "05_measurements" / "objects.csv")
    measurements = json.loads((args.results / "05_measurements" / "summary.json").read_text())
    if "dna_profile" not in measurements:
        raise SystemExit("results/05_measurements was made by an older version of step 5; "
                         "rerun scripts/05_measurements.py first.")
    dp, high = measurements["dna_profile"], measurements["thresholds"]["high"]
    s = test["summary"]
    fig = plt.figure(figsize=(W, H))

    # ---- A. The test image where the two methods differ most ------------------
    wide = per_image.pivot(index="name", columns="method", values="af1")
    name = (wide["cellpose"] - wide["baseline"]).idxmax()
    sample = {x.name: x for x in find_samples(args.data)}[name]
    img, gt = load_image(sample.image_path), load_mask(sample.mask_path)
    y0, x0 = busiest_window(gt, CROP)
    crop = (slice(y0, y0 + CROP), slice(x0, x0 + CROP))
    panels = [(f"Annotated outlines ({int(gt.max())} in the full image)", annotated_rgb(img, gt))]
    for method in ("baseline", "cellpose"):
        pred = tifffile.imread(args.results / "04_test" / "masks" / method / f"{name}.tif")
        ev = evaluate(gt, pred)
        panels.append((f"{NAMES[method]} ({ev.n_pred} found)", error_rgb(img, ev)))
    side_h = 0.30                      # crop height as a fraction of figure height
    side_w = side_h * H / W            # same physical size, so the crops are square
    for i, (label, rgb) in enumerate(panels):
        ax = fig.add_axes([0.04 + i * (side_w + 0.012), 0.535, side_w, side_h])
        ax.imshow(rgb[crop])
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(RULE)
        ax.set_xlabel(label, fontsize=9.5, color=INK, labelpad=5)
    heading(fig, 0.04, 0.855, "A", "Where the two methods differ most (crop of one test image)")
    keys = ["matched", "merge", "split", "missed", "extra"]
    fig.legend(handles=[Patch(color=ERROR_COLOURS[k], label=ERROR_NAMES[k]) for k in keys],
               loc="upper left", bbox_to_anchor=(0.035, 0.495), ncol=5, fontsize=8.5, frameon=False,
               handlelength=1.2, columnspacing=1.2)

    # ---- B. Against published methods ---------------------------------------------
    ax = fig.add_axes([0.815, 0.535, 0.17, side_h])
    rows = [(k, v, "published") for k, v in test["published_average_f1"].items()]
    rows += [("Baseline v2 (this project)", s["baseline"]["af1"], "baseline"),
             ("Cellpose-SAM* (this project)", s["cellpose"]["af1"], "cellpose")]
    rows.sort(key=lambda r: r[1])
    lo = min(v for _, v, _ in rows) - 0.03
    for y, (label, value, kind) in enumerate(rows):
        ours = kind != "published"
        ax.plot([lo, value], [y, y], color=RULE, lw=1, zorder=1)
        ax.scatter([value], [y], s=110 if ours else 70, color=COLOURS[kind] if ours else PUBLISHED_GREY,
                   edgecolor="white", linewidth=2, zorder=3)
        ax.annotate(f"{value:.3f}", (value, y), xytext=(9, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK, fontweight="bold" if ours else "normal")
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=9)
    for tick, (_, _, kind) in zip(ax.get_yticklabels(), rows):
        tick.set_fontweight("normal" if kind == "published" else "bold")
    ax.set_xlim(lo, max(v for _, v, _ in rows) + 0.04)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel("Average F1 (IoU 0.50–0.90)", fontsize=9, color=INK)
    ax.grid(axis="x", color=RULE, lw=0.8)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    style(ax)
    heading(fig, 0.695, 0.855, "B", "Against published methods, same test images")
    fig.text(0.695, 0.49, "* has most likely seen these images in training\nPublished results: Caicedo et al. 2019",
             fontsize=8, color=MUTED, va="top", linespacing=1.4)

    # ---- C. F1 as the outline requirement gets stricter ---------------------------
    ax = fig.add_axes([0.065, 0.085, 0.405, 0.3])
    curves = {}
    for method, g in per_image.groupby("method", sort=False):
        curves[method] = np.array([np.nanmean(f1_from_counts(g[f"tp@{t:.2f}"], g["n_gt"], g["n_pred"]))
                                   for t in IOU_THRESHOLDS])
        ax.plot(IOU_THRESHOLDS, curves[method], color=COLOURS[method], lw=2.2, marker="o", ms=6)
        ax.annotate(NAMES[method], (IOU_THRESHOLDS[-1], curves[method][-1]), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=9.5, color=INK)
    low = min(c.min() for c in curves.values())
    ax.set_ylim(min(0.5, np.floor(low * 10) / 10 - 0.05), 1.0)
    ax.set_xticks(IOU_THRESHOLDS[::2])
    ax.set_xlim(IOU_THRESHOLDS[0] - 0.015, IOU_THRESHOLDS[-1] + 0.1)
    ax.set_xlabel("IoU a nucleus needs with its annotated outline to count as correct", fontsize=9, color=INK)
    ax.set_ylabel("F1, mean over test images", fontsize=9, color=INK)
    ax.grid(axis="y", color=RULE, lw=0.8)
    style(ax)
    b, c = curves["baseline"], curves["cellpose"]
    takeaway = (min(b[0], c[0]) >= 0.85) and (c[-1] - b[-1] > c[0] - b[0])
    heading(fig, 0.04, 0.42, "C",
            "Both detect nuclei well; Cellpose's outlines hold up better" if takeaway
            else "F1 as the outline requirement gets stricter")

    # ---- D. DNA content ----------------------------------------------------------
    ax = fig.add_axes([0.545, 0.085, 0.44, 0.3])
    bins = np.arange(0, 3.01, 0.05)
    for source in ("annotated", "baseline", "cellpose"):
        v = objects.loc[objects["source"] == source, "dna_relative"].to_numpy()
        ax.hist(np.clip(v, bins[0], bins[-1]), bins=bins, histtype="step", lw=2, color=COLOURS[source],
                label=NAMES[source])
    for x, text in ((1.0, "median"), (2.0, "2× median")):
        ax.axvline(x, color=MUTED, lw=1)
        ax.annotate(text, (x, 1), xycoords=("data", "axes fraction"), xytext=(4, -12),
                    textcoords="offset points", fontsize=8.5, color=MUTED)
    b = dp["baseline"]
    note = (f"High-DNA objects (> {high}× the median), share of whole objects:\n"
            f"annotated {dp['annotated']['high_share_of_whole']:.1%}   "
            f"Cellpose {dp['cellpose']['high_share_of_whole']:.1%}   baseline {b['high_share_of_whole']:.1%}\n"
            f"{b['high_merges']} of the baseline's are merges of 2+ annotated nuclei;\n"
            f"without them its share is {b['high_share_without_merges']:.1%}")
    ax.text(0.98, 0.6, note, transform=ax.transAxes, ha="right", va="top", fontsize=8.8, color=INK,
            linespacing=1.5, bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=RULE))
    ax.set_xlabel("DNA content per object, relative to the image's median nucleus (last bin: 3 or more)",
                  fontsize=9, color=INK)
    ax.set_ylabel("Objects", fontsize=9, color=INK)
    ax.legend(frameon=False, fontsize=9, loc="upper right", bbox_to_anchor=(1.0, 0.92))  # below the line labels
    ax.grid(axis="y", color=RULE, lw=0.8)
    style(ax)
    merge_story = b["high_share_of_whole"] > dp["annotated"]["high_share_of_whole"] and b["high_merges"] > 0
    heading(fig, 0.52, 0.42, "D", "Merges add to the high-DNA tail" if merge_story
            else "DNA content from each set of outlines")

    # ---- Title and footer -----------------------------------------------------------
    fig.text(0.04, 0.955, "Nuclei segmentation on BBBC039: a classical pipeline vs Cellpose-SAM",
             fontsize=17, fontweight="bold", color=INK)
    cellprofiler = [v for k, v in test["published_average_f1"].items() if k.startswith("CellProfiler")]
    versus = ("above both CellProfiler pipelines in the benchmark paper" if s["baseline"]["af1"] > max(cellprofiler)
              else "compared with published pipelines in the panel on the right")
    fig.text(0.04, 0.905,
             f"A tuned threshold-and-watershed pipeline scores {s['baseline']['af1']:.3f} average F1, {versus}. "
             f"Cellpose-SAM scores {s['cellpose']['af1']:.3f}\n"
             f"on images it has most likely seen in training. Both were scored once on the 50-image test split.",
             fontsize=10.5, color=INK, linespacing=1.5)
    footer = "Data: BBBC039 (Caicedo et al.), Broad Bioimage Benchmark Collection, CC0"
    if args.repo:
        footer += f"   ·   Code: {args.repo}"
    fig.text(0.04, 0.018, footer, fontsize=8.5, color=MUTED)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Saved {args.out}  (image shown in panel A: {name})")


if __name__ == "__main__":
    main()
