"""Display helpers. These change how images look on screen, never the data."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from skimage.color import gray2rgb, label2rgb
from skimage.segmentation import find_boundaries


def stretch(img: np.ndarray, low: float = 1.0, high: float = 99.8) -> np.ndarray:
    """Percentile contrast stretch to 0-1 for display.

    A 16-bit image that only uses values up to ~4000 looks black if shown
    on the full 0-65535 scale, so we map the 1st-99.8th percentile to 0-1.
    """
    lo, hi = np.percentile(img, [low, high])
    out = (img.astype(np.float64) - lo) / max(hi - lo, 1e-9)
    return np.clip(out, 0.0, 1.0)


def outline_overlay(img: np.ndarray, labels: np.ndarray, colour=(1.0, 0.25, 0.25)) -> np.ndarray:
    """Draw each object's border on the image. 'inner' mode keeps a visible
    line between touching nuclei, which is exactly where errors hide."""
    rgb = gray2rgb(stretch(img))
    rgb[find_boundaries(labels, mode="inner")] = colour
    return rgb


ERROR_COLOURS = {
    "matched": (0.20, 0.80, 0.35),   # green outline: correct at IoU 0.5
    "merge": (0.92, 0.25, 0.20),     # red fill: one prediction covering several nuclei
    "split": (0.98, 0.62, 0.10),     # orange fill: fragment of a nucleus cut in pieces
    "extra": (0.85, 0.30, 0.85),     # magenta fill: prediction where there is no nucleus
    "poor_fit": (0.95, 0.90, 0.20),  # yellow outline: overlaps a nucleus, IoU <= 0.5
    "missed": (0.25, 0.55, 1.00),    # blue fill: true nucleus with nothing on it
}


def _blend(rgb: np.ndarray, where: np.ndarray, colour, alpha: float = 0.45) -> None:
    rgb[where] = (1 - alpha) * rgb[where] + alpha * np.asarray(colour)


ERROR_NAMES = {"matched": "correct", "poor_fit": "poor outline (IoU ≤ 0.5)", "merge": "merge",
               "split": "split fragment", "extra": "extra object", "missed": "missed nucleus"}


def annotated_rgb(img: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """The image with the annotated outlines in cyan."""
    rgb = gray2rgb(stretch(img))
    rgb[find_boundaries(gt, mode="inner")] = (0.3, 0.9, 1.0)
    return rgb


def error_rgb(img: np.ndarray, ev) -> np.ndarray:
    """The image with the prediction coloured by error type (ev: an ImageEval)."""
    rgb = gray2rgb(stretch(img))
    pred_status = np.concatenate([[""], ev.pred_status])[ev.pred]  # status per pixel
    gt_status = np.concatenate([[""], ev.gt_status])[ev.gt]
    for kind in ("merge", "split", "extra"):
        _blend(rgb, pred_status == kind, ERROR_COLOURS[kind])
    _blend(rgb, (gt_status == "missed") & (ev.pred == 0), ERROR_COLOURS["missed"], alpha=0.6)
    edges = find_boundaries(ev.pred, mode="inner")
    for kind in ERROR_COLOURS:
        rgb[edges & (pred_status == kind)] = ERROR_COLOURS[kind]
    return rgb


def save_error_overlay(img: np.ndarray, ev, title: str, out_path: Path) -> None:
    """Two panels: ground-truth outlines, and the prediction coloured by error type.

    `ev` is an ImageEval from nuclei.evaluate.
    """
    from matplotlib.patches import Patch

    left, right = annotated_rgb(img, ev.gt), error_rgb(img, ev)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), constrained_layout=True)
    axes[0].imshow(left)
    axes[0].set_title(f"Ground truth (n = {ev.n_gt})")
    axes[1].imshow(right)
    axes[1].set_title(f"Prediction (n = {ev.n_pred}), coloured by error at IoU 0.5")
    for ax in axes:
        ax.set_axis_off()
    handles = [Patch(color=ERROR_COLOURS[k], label=v) for k, v in ERROR_NAMES.items()]
    fig.legend(handles=handles, loc="outside lower center", ncol=6, fontsize=9, frameon=False)
    fig.suptitle(title, fontsize=10)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def save_overlay(img: np.ndarray, labels: np.ndarray, title: str, out_path: Path,
                 what: str = "Ground-truth labels") -> None:
    """Three panels: image, label map, outlines on image."""
    n = len(np.unique(labels[labels > 0]))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    axes[0].imshow(stretch(img), cmap="gray")
    axes[0].set_title("Image (contrast-stretched)")
    axes[1].imshow(label2rgb(labels, bg_label=0))
    axes[1].set_title(f"{what} (n = {n})")
    axes[2].imshow(outline_overlay(img, labels))
    axes[2].set_title("Outlines on image")
    for ax in axes:
        ax.set_axis_off()
    fig.suptitle(title, fontsize=10)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
