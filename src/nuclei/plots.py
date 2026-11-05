"""Result figures shared by steps 4 and 5. One look across all of them:
each method keeps its colour, the annotated reference is dark grey."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nuclei.evaluate import IOU_THRESHOLDS, SIZE_LABELS, f1_from_counts

COLOURS = {"baseline": "#2a78d6", "cellpose": "#eb6834", "annotated": "#3d3d3a"}
NAMES = {"baseline": "Baseline v2", "cellpose": "Cellpose-SAM", "annotated": "Annotated outlines"}
MARKERS = {"baseline": "o", "cellpose": "s", "annotated": "D"}
INK, MUTED, RULE = "#3d3d3a", "#77766f", "#e4e3dd"


def _style(ax) -> None:
    ax.grid(axis="y", color=RULE, lw=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=INK)


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def f1_by_iou(per_image: pd.DataFrame, path: Path) -> None:
    """F1 at each IoU threshold, averaged over images, one line per method."""
    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    for method, g in per_image.groupby("method", sort=False):
        f1 = np.array([np.nanmean(f1_from_counts(g[f"tp@{t:.2f}"], g["n_gt"], g["n_pred"]))
                       for t in IOU_THRESHOLDS])
        ax.plot(IOU_THRESHOLDS, f1, color=COLOURS[method], lw=2, marker=MARKERS[method], ms=7,
                label=f"{NAMES[method]} (average {f1.mean():.3f})")
        ax.annotate(NAMES[method], (IOU_THRESHOLDS[-1], f1[-1]), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=9, color=INK)
    ax.set_xticks(IOU_THRESHOLDS[::2])
    ax.set_xlabel("IoU threshold for a correct nucleus")
    ax.set_ylabel("F1 (mean over test images)")
    ax.set_ylim(0, 1)
    ax.set_xlim(IOU_THRESHOLDS[0] - 0.02, IOU_THRESHOLDS[-1] + 0.12)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax.set_title("F1 across IoU thresholds, test split", fontsize=10, loc="left")
    _style(ax)
    _save(fig, path)


def recall_by_size_bars(tables: dict[str, pd.DataFrame], path: Path) -> None:
    """Grouped bars: recall at IoU 0.5 per nucleus-size bin, one colour per method."""
    fig, ax = plt.subplots(figsize=(7.2, 4.0), constrained_layout=True)
    n = len(tables)
    width = min(0.8 / n, 0.34)
    x = np.arange(len(SIZE_LABELS))
    for i, (method, t) in enumerate(tables.items()):
        offset = (i - (n - 1) / 2) * (width + 0.02)
        bars = ax.bar(x + offset, t["recall"].to_numpy(), width=width, color=COLOURS[method],
                      label=NAMES[method])
        for bar, value in zip(bars, t["recall"].to_numpy()):
            ax.annotate(f"{value:.0%}", (bar.get_x() + bar.get_width() / 2, value), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=8, color=INK)
    counts = next(iter(tables.values()))["nuclei"].to_numpy()
    ax.set_xticks(x, [f"{lab} px\n(n = {c})" for lab, c in zip(SIZE_LABELS, counts)])
    ax.set_ylabel("Recall at IoU 0.5")
    ax.set_ylim(0, 1.12)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_title("Nuclei found, by annotated area", fontsize=10, loc="left")
    _style(ax)
    _save(fig, path)


def count_scatter(per_image: pd.DataFrame, path: Path) -> None:
    """Predicted vs annotated nucleus count per image."""
    fig, ax = plt.subplots(figsize=(5.2, 5.0), constrained_layout=True)
    top = max(per_image["n_gt"].max(), per_image["n_pred"].max()) * 1.05
    ax.plot([0, top], [0, top], color=MUTED, lw=1)
    ax.annotate("perfect count", (top * 0.72, top * 0.72), xytext=(4, -12), textcoords="offset points",
                fontsize=8, color=MUTED, rotation=45)
    for method, g in per_image.groupby("method", sort=False):
        ax.scatter(g["n_gt"], g["n_pred"], s=64, color=COLOURS[method], marker=MARKERS[method],
                   edgecolor="white", linewidth=1.5, label=NAMES[method], zorder=3)
    ax.set_xlabel("Annotated nuclei per image")
    ax.set_ylabel("Predicted nuclei per image")
    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_title("Nucleus counts per test image", fontsize=10, loc="left")
    _style(ax)
    _save(fig, path)


def step_histograms(values: dict[str, np.ndarray], bins: np.ndarray, xlabel: str, title: str,
                    path: Path, markers: dict[float, str] | None = None) -> None:
    """Overlaid outline histograms (no fills, so none hides another)."""
    fig, ax = plt.subplots(figsize=(7.2, 4.0), constrained_layout=True)
    for source, v in values.items():
        ax.hist(np.clip(v, bins[0], bins[-1]), bins=bins, histtype="step", lw=2,
                color=COLOURS[source], label=f"{NAMES[source]} (n = {len(v):,})")
    for x, text in (markers or {}).items():
        ax.axvline(x, color=MUTED, lw=1)
        ax.annotate(text, (x, 1), xycoords=("data", "axes fraction"), xytext=(4, -12),
                    textcoords="offset points", fontsize=8, color=MUTED)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Objects")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.set_title(title, fontsize=10, loc="left")
    _style(ax)
    _save(fig, path)
