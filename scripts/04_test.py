"""Step 4: score both frozen methods on the TEST split, once.

No decision so far has used the test split. This script reads the frozen
settings saved by steps 2 and 3, runs each method once on the 50 test
images, and saves scores, figures and the predicted masks (step 5 measures
nuclei from those masks without rerunning the models).

It refuses to run a second time. A test split is worth something only if
its numbers are produced once, after everything is frozen. --force exists
to reproduce the same numbers (another machine, a lost file), never to try
again with different settings.

Reminder: Cellpose has most likely been trained on BBBC039 images, test
split included, so its test score is a ceiling on seen data.

Run from PyCharm (right-click > Run) or:  python scripts/04_test.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import tifffile

from nuclei import plots
from nuclei.baseline import BaselineParams, segment
from nuclei.evaluate import evaluate, recall_by_size, summarize
from nuclei.io import find_samples, load_image, load_mask
from nuclei.viz import save_error_overlay

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SHOW = ["af1", "f1_50", "precision_50", "recall_50", "area_bias", "merges", "splits", "missed", "extra",
        "mean_abs_count_error"]

# Average F1 on the same 50 test images, from Caicedo et al. 2019 (Cytometry A).
PUBLISHED = {
    "CellProfiler, basic": 0.790,
    "CellProfiler, advanced": 0.811,
    "Random forest": 0.840,
    "DeepCell": 0.858,
    "U-Net": 0.898,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", type=Path, default=PROJECT_ROOT / "data" / "raw")
    p.add_argument("--out", type=Path, default=PROJECT_ROOT / "results" / "04_test")
    p.add_argument("--baseline-params", type=Path,
                   default=PROJECT_ROOT / "results" / "02_baseline_v2" / "params.json")
    p.add_argument("--cellpose-params", type=Path,
                   default=PROJECT_ROOT / "results" / "03_cellpose" / "params.json")
    p.add_argument("--force", action="store_true", help="overwrite existing test results (see docstring)")
    p.add_argument("--cpu", action="store_true", help="run Cellpose without the GPU")
    return p.parse_args()


def run_method(method: str, predict, samples, out: Path):
    """Predict, time and score every test image; save each predicted mask."""
    rows, objects, seconds = [], [], []
    (out / "masks" / method).mkdir(parents=True, exist_ok=True)
    for i, s in enumerate(samples, 1):
        img, gt = load_image(s.image_path), load_mask(s.mask_path)
        t0 = time.perf_counter()
        pred = predict(img)
        seconds.append(time.perf_counter() - t0)
        tifffile.imwrite(out / "masks" / method / f"{s.name}.tif", pred.astype(np.uint16))
        ev = evaluate(gt, pred)
        rows.append(dict(method=method, name=s.name, seconds=seconds[-1], **ev.counts()))
        objects.append(ev.objects().assign(method=method, name=s.name))
        if i % 10 == 0 or i == len(samples):
            print(f"  {method}: {i}/{len(samples)} images", flush=True)
    return pd.DataFrame(rows), pd.concat(objects, ignore_index=True)


def main() -> None:
    args = parse_args()
    if (args.out / "summary.json").exists() and not args.force:
        raise SystemExit(f"Test results already exist in {args.out}. The test split is scored once; "
                         "see the docstring before using --force.")
    args.out.mkdir(parents=True, exist_ok=True)

    baseline_params = BaselineParams(**json.loads(args.baseline_params.read_text()))
    cp = json.loads(args.cellpose_params.read_text())
    print(f"Baseline (frozen): {baseline_params}")
    print(f"Cellpose (frozen): model {cp['model']}, flow_threshold {cp['flow_threshold']}, "
          f"cellprob_threshold {cp['cellprob_threshold']}, min_size {cp['min_size']}")

    test = [s for s in find_samples(args.data) if s.split == "test" and s.mask_path is not None]
    print(f"\nTest split: {len(test)} images")

    # ---- 1. Run both methods once --------------------------------------------
    base_scores, base_objects = run_method("baseline", lambda img: segment(img, baseline_params), test, args.out)

    from nuclei.pretrained import CellposeRunner, environment_info
    env = environment_info()
    runner = CellposeRunner(cp["model"], gpu=not args.cpu)
    settings = dict(flow_threshold=cp["flow_threshold"], cellprob_threshold=cp["cellprob_threshold"],
                    min_size=cp["min_size"])
    cp_scores, cp_objects = run_method("cellpose", lambda img: runner.segment(img, **settings), test, args.out)
    runner.release()

    per_image = pd.concat([base_scores, cp_scores], ignore_index=True)
    objects = pd.concat([base_objects, cp_objects], ignore_index=True)
    per_image.to_csv(args.out / "per_image.csv", index=False)
    objects.to_csv(args.out / "objects.csv", index=False)

    # ---- 2. Scores --------------------------------------------------------------
    summary = {m: summarize(g) for m, g in per_image.groupby("method", sort=False)}
    speed = {m: float(g["seconds"].median()) for m, g in per_image.groupby("method", sort=False)}
    device = "CPU" if args.cpu else env.get("gpu", "CPU")
    print(f"\nTest split (merges, splits, missed, extra are totals over {len(test)} images):")
    print(pd.DataFrame({plots.NAMES[m]: pd.Series(s)[SHOW] for m, s in summary.items()})
          .astype(float).round(3).to_string())

    print("\nAverage F1 on the BBBC039 test split:")
    table = {**PUBLISHED, "Baseline v2 (this project)": summary["baseline"]["af1"],
             "Cellpose-SAM, seen these images (this project)": summary["cellpose"]["af1"]}
    for name, value in sorted(table.items(), key=lambda kv: kv[1]):
        print(f"  {value:.3f}  {name}")

    print("\nRecall at IoU 0.5 and area bias by nucleus area:")
    sizes = {m: recall_by_size(g) for m, g in objects.groupby("method", sort=False)}
    print(pd.concat({plots.NAMES[m]: t[["nuclei", "recall", "area_bias"]] for m, t in sizes.items()}, axis=1)
          .round(3).to_string())

    print(f"\nSpeed (median per image): baseline {speed['baseline']:.2f} s on one CPU core, "
          f"Cellpose {speed['cellpose']:.2f} s on {device}")

    # ---- 3. Figures -------------------------------------------------------------
    plots.f1_by_iou(per_image, args.out / "f1_by_iou.png")
    plots.recall_by_size_bars(sizes, args.out / "recall_by_size.png")

    wide = per_image.pivot(index="name", columns="method", values="af1")
    gap_name = (wide["cellpose"] - wide["baseline"]).idxmax()
    by_name = {s.name: s for s in test}
    picks = []
    for method, g in per_image.groupby("method", sort=False):
        ranked = g.sort_values("af1").reset_index(drop=True)
        picks += [(method, "worst", ranked.iloc[0]["name"]), (method, "median", ranked.iloc[len(ranked) // 2]["name"])]
    picks += [("baseline", "largest_gap", gap_name), ("cellpose", "largest_gap", gap_name)]
    for method, tag, name in picks:
        s = by_name[name]
        img = load_image(s.image_path)
        pred = tifffile.imread(args.out / "masks" / method / f"{name}.tif")
        ev = evaluate(load_mask(s.mask_path), pred)
        c = ev.counts()
        save_error_overlay(
            img, ev,
            f"{plots.NAMES[method]}, test, {tag.replace('_', ' ')}: {name}   AF1 = {c['af1']:.3f}, "
            f"merges {c['merges']}, splits {c['splits']}, missed {c['missed']}, extra {c['extra']}",
            args.out / "overlays" / f"{method}_{tag}_{name}.png",
        )

    (args.out / "summary.json").write_text(json.dumps({
        "baseline_params": baseline_params.as_dict(), "cellpose_params": cp, "environment": env,
        "summary": summary, "median_seconds_per_image": speed, "published_average_f1": PUBLISHED,
    }, indent=2))
    print(f"\nSaved scores, masks, figures and overlays to {args.out}")


if __name__ == "__main__":
    main()
