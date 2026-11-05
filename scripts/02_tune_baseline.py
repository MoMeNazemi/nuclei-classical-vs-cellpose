"""Step 2 (version 2): tune the classical baseline on TRAINING, check it on VALIDATION.

Changes from the first version, each prompted by its results:
- Settings are chosen by the benchmark paper's average F1 (IoU 0.50-0.90),
  which also rewards accurate outlines, instead of F1 at IoU 0.5 alone.
- Distance-transform seeds use a tie-safe peak test (see baseline.py).
- Also tried: a size-relative seed rule, brightness-based seeds, a threshold
  correction factor, and a guard that returns nothing for empty fields.

Nothing here reads the test split. Its scores are computed once, by step 4,
after both the baseline and the pretrained model are frozen.

Run from PyCharm (right-click > Run) or:  python scripts/02_tune_baseline.py
Uses several CPU cores; add --workers 1 to run on one (slower, easier to debug).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from nuclei import tuning
from nuclei.baseline import V1_PARAMS, BaselineParams, segment
from nuclei.evaluate import evaluate, recall_by_size, summarize
from nuclei.io import find_samples, load_image, load_mask
from nuclei.viz import save_error_overlay, save_overlay

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# sigma = 1 is fixed: in the first version every top result used it.
FOREGROUND = [(method, factor) for method in ("otsu", "li") for factor in (0.8, 1.0, 1.25)]
SEEDS = ([("shape", h) for h in (0.5, 1.0, 2.0)]
         + [("shape_rel", r) for r in (0.1, 0.2, 0.3)]
         + [("intensity", s) for s in (2.0, 4.0, 6.0)])
GRID = [BaselineParams(sigma=1.0, threshold=m, factor=f, seeds=sd, seed_value=v)
        for m, f in FOREGROUND for sd, v in SEEDS]

KEYS = ["threshold", "factor", "seeds", "seed_value"]
SHOW = ["af1", "f1_50", "precision_50", "recall_50", "area_bias", "merges", "splits", "missed", "extra"]
SEED_AXES = {"shape": "h (px)", "shape_rel": "r (fraction of peak height)", "intensity": "blur sigma (px)"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", type=Path, default=PROJECT_ROOT / "data" / "raw")
    p.add_argument("--out", type=Path, default=PROJECT_ROOT / "results" / "02_baseline_v2")
    p.add_argument("--workers", type=int, default=tuning.default_workers())
    return p.parse_args()


def summarize_by_config(scores: pd.DataFrame) -> pd.DataFrame:
    cols = list(BaselineParams().as_dict())
    rows = [dict(zip(cols, key), **summarize(g)) for key, g in scores.groupby(cols)]
    return pd.DataFrame(rows).sort_values(["af1", "f1_50"], ascending=False).reset_index(drop=True)


def params_from_row(row: pd.Series) -> BaselineParams:
    fields = BaselineParams().as_dict()
    return BaselineParams(**{k: type(v)(row[k]) for k, v in fields.items()})


def plot_seed_strategies(grid: pd.DataFrame, best: BaselineParams, out_path: Path) -> None:
    """For the chosen threshold, how each seed strategy and its setting perform."""
    g = grid[(grid["threshold"] == best.threshold) & (grid["factor"] == best.factor)]
    blue, orange, ink, rule = "#2a78d6", "#eb6834", "#3d3d3a", "#e4e3dd"
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.2), sharey="row", constrained_layout=True)
    for col, (method, xlabel) in enumerate(SEED_AXES.items()):
        d = g[g["seeds"] == method].sort_values("seed_value")
        top, bottom = axes[0, col], axes[1, col]
        top.plot(d["seed_value"], d["af1"], color=blue, lw=2, marker="o", ms=7)
        top.set_title(f"{method}: average F1", fontsize=10, loc="left")
        bottom.plot(d["seed_value"], d["merges"], color=blue, lw=2, marker="o", ms=7, label="merges")
        bottom.plot(d["seed_value"], d["splits"], color=orange, lw=2, marker="s", ms=7, label="splits")
        bottom.set_title(f"{method}: merges and splits", fontsize=10, loc="left")
        bottom.set_xlabel(xlabel)
        last = d.iloc[-1]
        upper = "merges" if last["merges"] >= last["splits"] else "splits"
        for name in ("merges", "splits"):
            bottom.annotate(name, (last["seed_value"], last[name]), xytext=(8, 7 if name == upper else -7),
                            textcoords="offset points", va="center", fontsize=9, color=ink)
        for ax in (top, bottom):
            if method == best.seeds:
                ax.axvline(best.seed_value, color="#9a9a94", lw=1, ls="--")
            ax.grid(axis="y", color=rule, lw=0.8)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(colors=ink)
            ax.margins(x=0.15)
    axes[1, 0].legend(frameon=False, fontsize=9)
    fig.suptitle(f"Seed strategies on the training split (threshold = {best.threshold} x {best.factor}); "
                 "dashed line = chosen", fontsize=10)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    samples = [s for s in find_samples(args.data) if s.mask_path is not None]
    train = [s for s in samples if s.split == "training"]
    val = [s for s in samples if s.split == "validation"]
    by_name = {s.name: s for s in samples}

    # ---- 1. Grid search on the training split ------------------------------
    configs = GRID + [V1_PARAMS]  # v1 rides along for comparison; it is not a candidate
    print(f"Tuning on {len(train)} training images: {len(GRID)} configurations "
          f"(+ version 1 for comparison), {args.workers} worker processes")
    scores, _ = tuning.run(train, configs, args.workers)
    scores.to_csv(args.out / "grid_training_per_image.csv", index=False)
    grid = summarize_by_config(scores)
    grid.to_csv(args.out / "grid_training.csv", index=False)
    candidates = grid[grid["seeds"] != "shape_v1"]

    print("\nTop configurations on training, by average F1 (IoU 0.50-0.90):")
    print(candidates[KEYS + SHOW].head(10).round(3).to_string(index=False))
    print("\nBest configuration for each seed strategy:")
    print(candidates.groupby("seeds", sort=False).head(1)[KEYS + SHOW].round(3).to_string(index=False))
    print("\nVersion 1 on training, scored the same way:")
    print(grid[grid["seeds"] == "shape_v1"][KEYS + SHOW].round(3).to_string(index=False))

    best = params_from_row(candidates.iloc[0])
    (args.out / "params.json").write_text(json.dumps(best.as_dict(), indent=2))
    print(f"\nChosen: {best}\n  -> saved to params.json (frozen from here on)")
    plot_seed_strategies(grid, best, args.out / "seed_strategies.png")

    # ---- 2. Validation: version 2 (chosen) and version 1, same images -------
    print(f"\nRunning version 2 and version 1 on {len(val)} validation images")
    val_scores, val_objects = tuning.run(val, [best, V1_PARAMS], args.workers, with_objects=True)
    v2 = val_scores[val_scores["seeds"] != "shape_v1"]
    v1 = val_scores[val_scores["seeds"] == "shape_v1"]
    v2.to_csv(args.out / "validation_per_image.csv", index=False)
    objects_v2 = val_objects[val_objects["seeds"] != "shape_v1"]
    objects_v2.to_csv(args.out / "validation_objects.csv", index=False)

    summary = {"version_2": summarize(v2), "version_1": summarize(v1),
               "version_2_training": candidates.iloc[0][SHOW].astype(float).to_dict()}
    (args.out / "validation_summary.json").write_text(json.dumps(summary, indent=2))
    table = pd.DataFrame({
        "v2 training": summary["version_2_training"],
        "v2 validation": pd.Series(summary["version_2"])[SHOW],
        "v1 validation": pd.Series(summary["version_1"])[SHOW],
    }).astype(float)
    print("\nVersion 2 on training and validation, and version 1 on validation")
    print("(merges, splits, missed, extra are totals; training has twice as many images):")
    print(table.round(3).to_string())

    print("\nValidation, version 2: recall at IoU 0.5 and area bias by nucleus area")
    print("(area bias +0.10 = predicted outlines enclose 10% more area than annotated):")
    print(recall_by_size(objects_v2).round(3).to_string())

    # ---- 3. Images that need a human look ------------------------------------
    # (a) No annotated nuclei. Empty field, or nobody annotated it? The preview
    #     answers that; version 1 shows what it found there before the guard.
    v1_found = dict(zip(v1["name"], v1["n_pred"]))
    train_v2 = scores[(scores[KEYS[:2]] == [best.threshold, best.factor]).all(axis=1)
                      & (scores["seeds"] == best.seeds) & (scores["seed_value"] == best.seed_value)]
    train_v1 = scores[scores["seeds"] == "shape_v1"]
    v1_found.update(dict(zip(train_v1["name"], train_v1["n_pred"])))
    looked_at = pd.concat([train_v2.assign(split="training"), v2.assign(split="validation")])
    empty = looked_at[looked_at["n_gt"] == 0]
    # (b) Nuclei annotated but nothing found: would show the guard misfiring.
    blank = looked_at[(looked_at["n_gt"] > 0) & (looked_at["n_pred"] == 0)]
    for title, rows, folder in (("Images with NO annotated nuclei", empty, "no_annotations"),
                                ("Images with annotated nuclei where version 2 found NOTHING", blank, "found_nothing")):
        if rows.empty:
            continue
        print(f"\n{title}:")
        for _, r in rows.iterrows():
            print(f"  {r['split']:10s} {r['name']}  annotated {r['n_gt']}, "
                  f"v2 found {r['n_pred']}, v1 found {v1_found.get(r['name'], '?')}")
            s = by_name[r["name"]]
            img = load_image(s.image_path)
            pred = segment(img, best) if r["n_pred"] > 0 else segment(img, V1_PARAMS)
            what = "Version 2 prediction" if r["n_pred"] > 0 else "Version 1 prediction"
            note = "" if r["n_pred"] > 0 else " (version 2 found nothing)"
            save_overlay(img, pred, f"{r['split']}: {r['name']}  [{title.lower()}]{note}",
                         args.out / folder / f"{r['name']}.png", what=what)
        print(f"  previews saved in {args.out / folder}")

    # ---- 4. Error overlays: worst, median and best validation images ---------
    ranked = v2[v2["n_gt"] > 0].sort_values("af1").reset_index(drop=True)
    picks = {"worst": 0, "median": len(ranked) // 2, "best": len(ranked) - 1}
    for tag, idx in picks.items():
        r = ranked.iloc[idx]
        s = by_name[r["name"]]
        img = load_image(s.image_path)
        ev = evaluate(load_mask(s.mask_path), segment(img, best))
        save_error_overlay(
            img, ev,
            f"validation, {tag} average F1: {r['name']}   AF1 = {r['af1']:.3f}, F1@0.5 = {r['f1_50']:.3f}, "
            f"merges {r['merges']}, splits {r['splits']}, missed {r['missed']}, extra {r['extra']}",
            args.out / "overlays" / f"{tag}_{r['name']}.png",
        )
    print(f"\nSaved tables, params.json, seed_strategies.png and overlays/ to {args.out}")


if __name__ == "__main__":
    main()
