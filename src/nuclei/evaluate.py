"""Object-level evaluation: compare a predicted label image with the ground truth.

Headline metrics follow the benchmark paper for this dataset (Caicedo et al.
2019, Cytometry A), so the numbers can be compared with its results:

- A predicted nucleus is correct at IoU threshold t if its intersection over
  union with a true nucleus is ABOVE t.
- F1 = 2TP / (2TP + FP + FN) is computed per image at t = 0.50, 0.55, ..., 0.90,
  averaged over images, then over thresholds: the paper's "average F1".
  Because it includes strict thresholds, it rewards accurate outlines, not
  only detection. F1 at IoU 0.5 is reported too, as the detection score.
- Ground-truth objects are not filtered by size, so micronuclei count.

Error categories are defined by coverage ("most of" = more than half):

- merge:  one prediction covers most of two or more true nuclei
- split:  one true nucleus contains most of two or more predictions
- missed: a true nucleus with no prediction belonging to it
- extra:  a prediction belonging to no true nucleus
(a prediction belongs to a nucleus if it covers most of the nucleus, or
most of the prediction lies inside the nucleus)

The authors' code instead flags merges and splits by IoU > 0.1. That rule
misses small fragments: a thin sliver cut out of a nucleus has IoU < 0.1 with
it. Their counts are still computed, as merges_ref and splits_ref, for
comparison.

One difference from the paper: an image where both the ground truth and the
prediction are empty is left unscored (NaN) instead of scoring F1 = 0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from skimage.segmentation import relabel_sequential

IOU_THRESHOLDS = np.round(np.arange(0.50, 0.91, 0.05), 2)  # 0.50 ... 0.90, as in the paper
MOSTLY = 0.5           # "most of" means more than this fraction of an object's pixels
REF_OVERLAP_IOU = 0.1  # the authors' merge/split rule, kept for comparison

# Area bins (px) for the size breakdown. 250 / 625 / 900 are the authors'
# bins; the extra 25 px edge separates micronucleus-sized objects.
SIZE_BINS = [0, 25, 250, 625, 900, np.inf]
SIZE_LABELS = ["<25", "25-250", "250-625", "625-900", ">900"]


def overlap_counts(gt: np.ndarray, pred: np.ndarray):
    """Pixel counts shared by every true object (rows) and predicted object (columns).

    Both inputs must be label images numbered 1..N with no gaps.
    Returns (intersection, area_gt, area_pred), background excluded.

    How it works: every pixel belongs to one (true label, predicted label)
    pair. Encoding each pair as one integer and counting them with bincount
    gives every intersection in a single pass over the image.
    """
    n_gt, n_pred = int(gt.max()), int(pred.max())
    pair = gt.astype(np.int64) * (n_pred + 1) + pred.astype(np.int64)
    counts = np.bincount(pair.ravel(), minlength=(n_gt + 1) * (n_pred + 1))
    table = counts.reshape(n_gt + 1, n_pred + 1)
    return table[1:, 1:], table.sum(axis=1)[1:], table.sum(axis=0)[1:]


def f1_from_counts(tp, n_gt, n_pred):
    """F1 = 2TP / (2TP + FP + FN). Since TP + FN = n_gt and TP + FP = n_pred,
    the denominator is simply n_gt + n_pred."""
    denom = n_gt + n_pred
    return np.where(denom > 0, 2 * np.asarray(tp) / np.maximum(denom, 1), np.nan)


@dataclass
class ImageEval:
    """Evaluation of one image. Labels are renumbered 1..N, so object k is
    row/column k-1 of the tables and entry k-1 of the status arrays."""

    gt: np.ndarray
    pred: np.ndarray
    iou: np.ndarray
    area_gt: np.ndarray
    area_pred: np.ndarray
    tp: np.ndarray            # true positives at each IOU_THRESHOLDS value
    merges: int
    splits: int
    merges_ref: int
    splits_ref: int
    gt_status: np.ndarray     # per true nucleus, at IoU 0.5
    pred_status: np.ndarray   # per predicted object, at IoU 0.5
    pred_covers: np.ndarray   # per predicted object: how many true nuclei it covers most of (2+ = merge)

    @property
    def n_gt(self) -> int:
        return self.iou.shape[0]

    @property
    def n_pred(self) -> int:
        return self.iou.shape[1]

    def matched_pairs(self):
        """(true index, predicted index) of every match at IoU 0.5."""
        g, p = np.nonzero(self.iou > 0.5)
        return g, p

    def counts(self) -> dict:
        """One row of numbers for a per-image table."""
        f1 = f1_from_counts(self.tp, self.n_gt, self.n_pred)
        g, p = self.matched_pairs()
        row = {
            "n_gt": self.n_gt,
            "n_pred": self.n_pred,
            "count_error": self.n_pred - self.n_gt,
            "tp_50": int(self.tp[0]),
            "fp_50": self.n_pred - int(self.tp[0]),
            "fn_50": self.n_gt - int(self.tp[0]),
            "f1_50": float(f1[0]),
            "af1": float(np.mean(f1)) if self.n_gt + self.n_pred > 0 else np.nan,
            "merges": self.merges,
            "splits": self.splits,
            "missed": int((self.gt_status == "missed").sum()),
            "extra": int((self.pred_status == "extra").sum()),
            "merges_ref": self.merges_ref,
            "splits_ref": self.splits_ref,
            # For the area bias: summed log of (predicted area / true area) over matches.
            "n_matched": len(g),
            "sum_log_area_ratio": float(np.log(self.area_pred[p] / self.area_gt[g]).sum()),
        }
        for t, tp in zip(IOU_THRESHOLDS, self.tp):
            row[f"tp@{t:.2f}"] = int(tp)  # raw counts, so results can be pooled across images
        return row

    def objects(self) -> pd.DataFrame:
        """One row per true nucleus: its area, best IoU and what happened to it."""
        best = self.iou.max(axis=1) if self.n_pred else np.zeros(self.n_gt)
        pred_area = np.full(self.n_gt, np.nan)
        g, p = self.matched_pairs()
        pred_area[g] = self.area_pred[p]
        return pd.DataFrame({
            "label": np.arange(1, self.n_gt + 1),
            "area": self.area_gt,
            "pred_area": pred_area,   # area of the matching prediction, if any
            "best_iou": best,
            "status": self.gt_status,
        })


def evaluate(gt: np.ndarray, pred: np.ndarray) -> ImageEval:
    """Compare a prediction with the ground truth for one image."""
    gt = relabel_sequential(gt)[0]
    pred = relabel_sequential(pred)[0]
    inter, area_gt, area_pred = overlap_counts(gt, pred)
    union = area_gt[:, None] + area_pred[None, :] - inter
    iou = inter / np.maximum(union, 1)
    cover = inter / np.maximum(area_gt[:, None], 1)     # share of each true nucleus covered by each prediction
    inside = inter / np.maximum(area_pred[None, :], 1)  # share of each prediction lying inside each true nucleus

    # Objects don't overlap within one image, so IoU > 0.5 can pair an object
    # with at most ONE partner: counting rows with a match counts true positives.
    tp = np.array([(iou > t).any(axis=1).sum() for t in IOU_THRESHOLDS])

    matched = iou > 0.5
    belongs = (cover > MOSTLY) | (inside > MOSTLY)
    covers = (cover > MOSTLY).sum(axis=0)       # per prediction: true nuclei it covers most of
    merge = covers >= 2
    split = (inside > MOSTLY).sum(axis=1) >= 2  # per true nucleus

    # What happened to each true nucleus. Later lines take precedence.
    gt_status = np.full(iou.shape[0], "poor_fit", dtype=object)  # has a prediction, but IoU <= 0.5
    gt_status[~belongs.any(axis=1)] = "missed"
    gt_status[split] = "split"
    gt_status[(cover[:, merge] > MOSTLY).any(axis=1)] = "merged"
    gt_status[matched.any(axis=1)] = "matched"

    # What each prediction is. Later lines take precedence.
    pred_status = np.full(iou.shape[1], "poor_fit", dtype=object)
    pred_status[~belongs.any(axis=0)] = "extra"
    pred_status[(inside[split, :] > MOSTLY).any(axis=0)] = "split"
    pred_status[matched.any(axis=0)] = "matched"
    pred_status[merge] = "merge"  # shown as a merge even if one nucleus inside still matched

    ref = iou > REF_OVERLAP_IOU
    return ImageEval(
        gt=gt, pred=pred, iou=iou, area_gt=area_gt, area_pred=area_pred, tp=tp,
        merges=int(merge.sum()), splits=int(split.sum()),
        merges_ref=int((ref.sum(axis=0) > 1).sum()), splits_ref=int((ref.sum(axis=1) > 1).sum()),
        gt_status=gt_status, pred_status=pred_status, pred_covers=covers,
    )


def summarize(per_image: pd.DataFrame) -> dict:
    """Combine a per-image table (rows from ImageEval.counts) into one summary.

    af1 and f1_50 average each image's own score, as the paper does (every
    image weighs the same). The *_pooled versions add up counts over all
    images first (every nucleus weighs the same).
    """
    n_gt = per_image["n_gt"].sum()
    n_pred = per_image["n_pred"].sum()
    tp = np.array([per_image[f"tp@{t:.2f}"].sum() for t in IOU_THRESHOLDS])
    pooled_f1 = f1_from_counts(tp, n_gt, n_pred)
    n_matched = per_image["n_matched"].sum()
    return {
        "images": len(per_image),
        "true_nuclei": int(n_gt),
        "predicted": int(n_pred),
        "af1": float(per_image["af1"].mean()),
        "f1_50": float(per_image["f1_50"].mean()),
        "af1_pooled": float(np.mean(pooled_f1)),
        "f1_50_pooled": float(pooled_f1[0]),
        "precision_50": float(tp[0] / n_pred) if n_pred else np.nan,
        "recall_50": float(tp[0] / n_gt) if n_gt else np.nan,
        # Typical size error of correctly found nuclei: +0.10 means predicted
        # outlines enclose 10% more area than the annotated ones.
        "area_bias": float(np.exp(per_image["sum_log_area_ratio"].sum() / n_matched) - 1) if n_matched else np.nan,
        "merges": int(per_image["merges"].sum()),
        "splits": int(per_image["splits"].sum()),
        "missed": int(per_image["missed"].sum()),
        "extra": int(per_image["extra"].sum()),
        "merges_ref": int(per_image["merges_ref"].sum()),
        "splits_ref": int(per_image["splits_ref"].sum()),
        "mean_abs_count_error": float(per_image["count_error"].abs().mean()),
    }


def recall_by_size(objects: pd.DataFrame) -> pd.DataFrame:
    """Recall at IoU 0.5 and area bias for true nuclei grouped by area."""
    size = pd.cut(objects["area"], SIZE_BINS, labels=SIZE_LABELS, right=False)
    found = objects["status"] == "matched"
    log_ratio = np.log(objects["pred_area"] / objects["area"])  # NaN where not matched
    out = pd.DataFrame({
        "nuclei": found.groupby(size, observed=False).size(),
        "found": found.groupby(size, observed=False).sum(),
        "area_bias": np.exp(log_ratio.groupby(size, observed=False).mean()) - 1,
    })
    out.insert(2, "recall", out["found"] / out["nuclei"].replace(0, np.nan))
    out.index.name = "area_px"
    return out
