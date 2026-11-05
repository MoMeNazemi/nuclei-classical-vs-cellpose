"""Step 3: a pretrained deep-learning model, Cellpose-SAM, on the same images.

Same protocol as step 2: the model and its two thresholds are chosen on
TRAINING, checked once on VALIDATION, then frozen. The test split stays shut.

Caveat: Cellpose's nuclei training data includes BBBC039 (157 of its images
directly, the rest through the Data Science Bowl 2018 set), and Cellpose-SAM
was trained on that nuclei data, minus a ~10% held-out test set. So Cellpose
has most likely seen most of these images and their annotations. Its scores
here show how well it reproduces data it was trained on, not how well it
generalises.

Run from PyCharm (right-click > Run) or:
    python scripts/03_tune_cellpose.py            full run (GPU strongly recommended)
    python scripts/03_tune_cellpose.py --limit 5  quick check on 5 images per split
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd

from nuclei.evaluate import evaluate, recall_by_size, summarize
from nuclei.io import find_samples, load_image, load_mask
from nuclei.pretrained import CELLPOSE_VERSION, DEFAULTS, CellposeRunner, environment_info
from nuclei.viz import save_error_overlay

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODELS = ["cpsam", "cpsam_v2"]  # original Cellpose-SAM (April 2025) and its June 2026 update
FLOW_THRESHOLDS = [0.4, 0.6, 0.8]
CELLPROB_THRESHOLDS = [-1.0, 0.0, 1.0]
DEFAULT = ("cpsam_v2", DEFAULTS["flow_threshold"], DEFAULTS["cellprob_threshold"])  # Cellpose 4.2 out of the box

KEYS = ["model", "flow_threshold", "cellprob_threshold"]
SHOW = ["af1", "f1_50", "precision_50", "recall_50", "area_bias", "merges", "splits", "missed", "extra"]
CAVEAT = ("NOTE: Cellpose's nuclei training data includes BBBC039, so these images were most likely\n"
          "seen in training. Read the scores as a ceiling on seen data, not as generalisation.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", type=Path, default=PROJECT_ROOT / "data" / "raw")
    p.add_argument("--out", type=Path, default=PROJECT_ROOT / "results" / "03_cellpose")
    p.add_argument("--baseline", type=Path, default=PROJECT_ROOT / "results" / "02_baseline_v2",
                   help="step 2 results folder, for the side-by-side comparison")
    p.add_argument("--limit", type=int, default=0, help="use only the first N images of each split (quick check)")
    p.add_argument("--cpu", action="store_true", help="don't use the GPU")
    return p.parse_args()


def run_model(runner: CellposeRunner, samples, settings, with_objects: bool = False):
    """Network once per image, then masks and scores for every threshold setting."""
    rows, objects, net_times = [], [], []
    start = time.time()
    for i, s in enumerate(samples, 1):
        img, gt = load_image(s.image_path), load_mask(s.mask_path)
        t0 = time.time()
        dP, cellprob = runner.network(img)
        net_times.append(time.time() - t0)
        for f, c in settings:
            t1 = time.time()
            pred = runner.masks(dP, cellprob, flow_threshold=f, cellprob_threshold=c)
            mask_time = time.time() - t1
            ev = evaluate(gt, pred)
            key = dict(model=runner.model_name, flow_threshold=f, cellprob_threshold=c)
            rows.append(dict(name=s.name, **key, **ev.counts(), net_s=net_times[-1], masks_s=mask_time))
            if with_objects:
                objects.append(ev.objects().assign(name=s.name, **key))
        if i % 10 == 0 or i == len(samples):
            elapsed = time.time() - start
            print(f"  {runner.model_name}: {i}/{len(samples)} images, {elapsed:.0f} s elapsed, "
                  f"~{elapsed / i * (len(samples) - i):.0f} s left", flush=True)
    objects = pd.concat(objects, ignore_index=True) if objects else None
    return pd.DataFrame(rows), objects


def self_check(runner: CellposeRunner, img: np.ndarray, gt: np.ndarray) -> None:
    """Masks rebuilt from cached network output should match Cellpose's own
    model.eval at default settings. (GPU arithmetic is not bit-exact between
    runs, so we compare scores, not pixels.)"""
    own = evaluate(gt, runner.segment(img)).counts()
    dP, cellprob = runner.network(img)
    ours = evaluate(gt, runner.masks(dP, cellprob)).counts()
    same_score = np.isclose(own["af1"], ours["af1"], atol=0.01, equal_nan=True)
    ok = bool(same_score) and abs(own["n_pred"] - ours["n_pred"]) <= 1
    print(f"  self-check ({runner.model_name}): Cellpose's own eval found {own['n_pred']} nuclei "
          f"(AF1 {own['af1']:.3f}), our two-stage route {ours['n_pred']} (AF1 {ours['af1']:.3f}) "
          f"-> {'OK' if ok else 'MISMATCH'}")
    if not ok:
        raise SystemExit("Self-check failed: masks rebuilt from the cached network output differ from "
                         "Cellpose's own result, so the threshold search below would not be trustworthy. "
                         f"Check that the installed Cellpose is version {CELLPOSE_VERSION}.")


def summarize_by_config(scores: pd.DataFrame) -> pd.DataFrame:
    rows = [dict(zip(KEYS, key), **summarize(g), net_s=g["net_s"].iloc[1:].mean() if len(g) > 1 else g["net_s"].mean())
            for key, g in scores.groupby(KEYS)]
    return pd.DataFrame(rows).sort_values(["af1", "f1_50"], ascending=False).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    env = environment_info()
    print(f"Cellpose {env['cellpose']}, PyTorch {env['torch']}, "
          f"GPU: {env.get('gpu', 'none (running on CPU, this will be slow)')}")
    if args.cpu:
        print("  --cpu given: not using the GPU")
    print(CAVEAT)

    samples = [s for s in find_samples(args.data) if s.mask_path is not None]
    train = [s for s in samples if s.split == "training"]
    val = [s for s in samples if s.split == "validation"]
    if args.limit:
        train, val = train[:args.limit], val[:args.limit]
    by_name = {s.name: s for s in samples}
    settings = list(itertools.product(FLOW_THRESHOLDS, CELLPROB_THRESHOLDS))

    # ---- 1. Every model and threshold setting on the training split ---------
    print(f"\nTraining split: {len(train)} images, {len(MODELS)} models x {len(settings)} threshold settings")
    parts = []
    for name in MODELS:
        runner = CellposeRunner(name, gpu=not args.cpu)
        for s in train:  # self-check on the first training image with annotated nuclei
            gt = load_mask(s.mask_path)
            if gt.max() > 0:
                self_check(runner, load_image(s.image_path), gt)
                break
        scores, _ = run_model(runner, train, settings)
        parts.append(scores)
        runner.release()
    scores = pd.concat(parts, ignore_index=True)
    scores.to_csv(args.out / "grid_training_per_image.csv", index=False)
    grid = summarize_by_config(scores)
    grid.to_csv(args.out / "grid_training.csv", index=False)

    print("\nTop settings on training, by average F1 (IoU 0.50-0.90):")
    print(grid[KEYS + SHOW].head(8).round(3).to_string(index=False))
    print("\nBest setting for each model, and the out-of-the-box default:")
    default_row = grid[(grid[KEYS] == list(DEFAULT)).all(axis=1)]
    print(pd.concat([grid.groupby("model", sort=False).head(1), default_row])
          [KEYS + SHOW + ["net_s"]].round(3).to_string(index=False))

    top = grid.iloc[0]
    best = (str(top["model"]), float(top["flow_threshold"]), float(top["cellprob_threshold"]))
    params = dict(zip(KEYS, best), min_size=DEFAULTS["min_size"], diameter=None, environment=env)
    (args.out / "params.json").write_text(json.dumps(params, indent=2))
    print(f"\nChosen: model {best[0]}, flow_threshold {best[1]}, cellprob_threshold {best[2]}"
          f"\n  -> saved to params.json (frozen from here on)")

    # ---- 2. Validation: the chosen setting and the default ------------------
    print(f"\nValidation split: {len(val)} images")
    val_parts, val_objects = [], []
    for name in dict.fromkeys([best[0], DEFAULT[0]]):  # each model once, chosen first
        wanted = [(f, c) for (m, f, c) in (best, DEFAULT) if m == name]
        runner = CellposeRunner(name, gpu=not args.cpu)
        part, objs = run_model(runner, val, list(dict.fromkeys(wanted)), with_objects=True)
        val_parts.append(part)
        val_objects.append(objs)
        runner.release()
    val_scores = pd.concat(val_parts, ignore_index=True)
    val_objects = pd.concat(val_objects, ignore_index=True)
    is_best = (val_scores[KEYS] == list(best)).all(axis=1)
    is_default = (val_scores[KEYS] == list(DEFAULT)).all(axis=1)
    val_scores[is_best].to_csv(args.out / "validation_per_image.csv", index=False)
    best_objects = val_objects[(val_objects[KEYS] == list(best)).all(axis=1)]
    best_objects.to_csv(args.out / "validation_objects.csv", index=False)

    summary = {"cellpose_chosen": summarize(val_scores[is_best]),
               "cellpose_default": summarize(val_scores[is_default])}
    columns = {"Cellpose chosen": pd.Series(summary["cellpose_chosen"])[SHOW],
               "Cellpose default": pd.Series(summary["cellpose_default"])[SHOW]}
    baseline_file = args.baseline / "validation_summary.json"
    if baseline_file.exists() and not args.limit:
        columns["baseline v2"] = pd.Series(json.loads(baseline_file.read_text())["version_2"])[SHOW]
    (args.out / "validation_summary.json").write_text(json.dumps(summary, indent=2))
    print("\nValidation split (merges, splits, missed, extra are totals):")
    print(pd.DataFrame(columns).astype(float).round(3).to_string())

    print("\nValidation, recall at IoU 0.5 and area bias by nucleus area:")
    size_table = {"Cellpose": recall_by_size(best_objects)}
    baseline_objects = args.baseline / "validation_objects.csv"
    if baseline_objects.exists() and not args.limit:
        size_table["baseline v2"] = recall_by_size(pd.read_csv(baseline_objects))
    print(pd.concat({k: v[["nuclei", "recall", "area_bias"]] for k, v in size_table.items()}, axis=1)
          .round(3).to_string())

    chosen = val_scores[is_best]
    empty = chosen[chosen["n_gt"] == 0]
    if len(empty):
        print("\nImages with no annotated nuclei, and what Cellpose found there:")
        for _, r in empty.iterrows():
            print(f"  {r['name']}: {r['n_pred']} objects")

    print(f"\nSpeed (median per image): {chosen['net_s'].median():.2f} s for the network, "
          f"{chosen['masks_s'].median():.2f} s to build masks ({'CPU' if args.cpu else env.get('gpu', 'CPU')})")

    # ---- 3. Error overlays: worst, median and best validation images ---------
    ranked = chosen[chosen["n_gt"] > 0].sort_values("af1").reset_index(drop=True)
    picks = {"worst": 0, "median": len(ranked) // 2, "best": len(ranked) - 1}
    runner = CellposeRunner(best[0], gpu=not args.cpu)
    for tag, idx in picks.items():
        r = ranked.iloc[idx]
        s = by_name[r["name"]]
        img = load_image(s.image_path)
        dP, cellprob = runner.network(img)
        ev = evaluate(load_mask(s.mask_path),
                      runner.masks(dP, cellprob, flow_threshold=best[1], cellprob_threshold=best[2]))
        save_error_overlay(
            img, ev,
            f"Cellpose ({best[0]}), validation, {tag} average F1: {r['name']}   AF1 = {r['af1']:.3f}, "
            f"F1@0.5 = {r['f1_50']:.3f}, merges {r['merges']}, splits {r['splits']}, "
            f"missed {r['missed']}, extra {r['extra']}",
            args.out / "overlays" / f"{tag}_{r['name']}.png",
        )
    runner.release()
    print(f"\nSaved tables, params.json and overlays/ to {args.out}")


if __name__ == "__main__":
    main()
