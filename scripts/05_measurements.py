"""Step 5: do the segmentation differences change what a biologist would measure?

From the masks step 4 saved for the test split, measure every object's area
and DNA content (background-subtracted summed Hoechst intensity) for three
sources: the annotated outlines (the reference), the baseline and Cellpose.

1. Counts: nuclei per image, compared with the annotation.
2. Per-nucleus agreement: how far a method's area and DNA content are from the
   annotated outline's, (a) over the nuclei each method found (IoU > 0.5), and
   (b) over the nuclei BOTH methods found, so the two are compared on the same
   nuclei.
3. Population: the distribution of DNA content on a relative scale (1 = the
   image's median annotated nucleus). A merge adds up the DNA of every nucleus
   it covers, so it lands in the high tail of that distribution, where G2/M
   cells would sit in an untreated culture. The scale is relative because
   BBBC039 shows a different compound in each image; see measure.py.

This step makes no decisions, so it can be rerun freely.

Run from PyCharm (right-click > Run) or:  python scripts/05_measurements.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import tifffile

from nuclei import plots
from nuclei.evaluate import evaluate
from nuclei.io import find_samples, load_image, load_mask
from nuclei.measure import area_and_dna, background_level, found_by_all, image_unit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METHODS = ["baseline", "cellpose"]
LOW, HIGH = 0.5, 1.6  # relative DNA content: below LOW = fragment-like, above HIGH = high-DNA object


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", type=Path, default=PROJECT_ROOT / "data" / "raw")
    p.add_argument("--test-results", type=Path, default=PROJECT_ROOT / "results" / "04_test")
    p.add_argument("--out", type=Path, default=PROJECT_ROOT / "results" / "05_measurements")
    return p.parse_args()


def describe(errors: np.ndarray) -> dict:
    return {
        "nuclei": int(len(errors)),
        "median error": float(np.median(errors)),
        "5th pct": float(np.percentile(errors, 5)),
        "95th pct": float(np.percentile(errors, 95)),
        "within ±10%": float(np.mean(np.abs(errors) <= 0.10)),
    }


def agreement(pairs: pd.DataFrame) -> dict:
    """Area and DNA-content error statistics per method."""
    out = {}
    for method, g in pairs.groupby("method", sort=False):
        out[method] = {"area": describe(g["area_error"].to_numpy()),
                       "DNA content": describe(g["dna_error"].to_numpy())}
    return out


def print_agreement(stats: dict) -> None:
    rows = {(plots.NAMES[m], label): d for m, by_label in stats.items() for label, d in by_label.items()}
    table = pd.DataFrame(rows).T
    table["nuclei"] = table["nuclei"].astype(int)
    print(table.round(3).to_string())


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    test = [s for s in find_samples(args.data) if s.split == "test" and s.mask_path is not None]
    if not test:
        raise SystemExit(f"No annotated test images found under {args.data}.")

    objects, pairs, counts = [], [], []
    for s in test:
        img, annotated = load_image(s.image_path), load_mask(s.mask_path)
        signal = img.astype(np.float64) - background_level(img, annotated)
        a_area, a_dna = area_and_dna(annotated, signal)
        unit = image_unit(a_area, a_dna)
        objects.append(pd.DataFrame({"source": "annotated", "name": s.name, "area": a_area, "dna": a_dna,
                                     "dna_relative": a_dna / unit, "status": "annotated", "covers": 1}))
        for method in METHODS:
            pred = tifffile.imread(args.test_results / "masks" / method / f"{s.name}.tif")
            ev = evaluate(annotated, pred)  # relabels both images 1..N, the same way for both methods
            p_area, p_dna = area_and_dna(ev.pred, signal)
            objects.append(pd.DataFrame({"source": method, "name": s.name, "area": p_area, "dna": p_dna,
                                         "dna_relative": p_dna / unit, "status": ev.pred_status,
                                         "covers": ev.pred_covers}))
            g_area, g_dna = area_and_dna(ev.gt, signal)
            g, p = ev.matched_pairs()
            keep = g_dna[g] > 0  # a relative error needs a positive reference
            g, p = g[keep], p[keep]
            pairs.append(pd.DataFrame({"method": method, "name": s.name, "gt": g,
                                       "area_error": p_area[p] / g_area[g] - 1,
                                       "dna_error": p_dna[p] / g_dna[g] - 1}))
            counts.append({"method": method, "name": s.name, "n_gt": ev.n_gt, "n_pred": ev.n_pred})

    objects = pd.concat(objects, ignore_index=True)
    pairs = pd.concat(pairs, ignore_index=True)
    counts = pd.DataFrame(counts)
    objects.to_csv(args.out / "objects.csv", index=False)
    pairs.to_csv(args.out / "matched_pairs.csv", index=False)
    summary = {"counts": {}, "agreement": {}, "agreement_found_by_both": {}, "dna_profile": {},
               "thresholds": {"low": LOW, "high": HIGH}}

    # ---- 1. Counts --------------------------------------------------------------
    print("Nucleus counts per test image vs annotated:")
    for method, g in counts.groupby("method", sort=False):
        pct = (g["n_pred"] - g["n_gt"]) / g["n_gt"]
        worst = g.loc[pct.abs().idxmax()]
        summary["counts"][method] = {
            "mean_abs_error": float(pct.abs().mean()), "worst_abs_error": float(pct.abs().max()),
            "worst_image": worst["name"], "worst_n_pred": int(worst["n_pred"]), "worst_n_gt": int(worst["n_gt"]),
            "total_pred": int(g["n_pred"].sum()), "total_gt": int(g["n_gt"].sum()),
        }
        print(f"  {plots.NAMES[method]:13s} mean |error| {pct.abs().mean():.1%}, total {g['n_pred'].sum()} "
              f"vs {g['n_gt'].sum()} annotated; worst image {worst['name']}: "
              f"{worst['n_pred']} found vs {worst['n_gt']} annotated")

    # ---- 2. Per-nucleus agreement -------------------------------------------------
    print("\n(relative error: +0.05 = the method measures 5% more than the annotated outline)")
    print("a) Nuclei each method found (IoU > 0.5), so the two methods are measured on different nuclei:")
    summary["agreement"] = agreement(pairs)
    print_agreement(summary["agreement"])
    both = found_by_all(pairs)
    print("\nb) Only the nuclei BOTH methods found, so the comparison is like for like:")
    summary["agreement_found_by_both"] = agreement(both)
    print_agreement(summary["agreement_found_by_both"])

    # ---- 3. Population: relative DNA content --------------------------------------
    # Shares over ALL objects mix two effects: missing tiny fragments shrinks a
    # method's total and inflates every other share. So high-DNA objects are
    # counted among "whole" objects (at least LOW on the relative scale).
    print(f"\nDNA content relative to the image's median annotated nucleus (= 1):")
    for source, g in objects.groupby("source", sort=False):
        v = g["dna_relative"].to_numpy()
        whole, high = v >= LOW, v > HIGH
        entry = {"objects": int(len(v)), "fragment_like": int((~whole).sum()), "whole": int(whole.sum()),
                 "high": int(high.sum()), "high_share_of_whole": float(high.sum() / max(whole.sum(), 1))}
        line = (f"  {plots.NAMES[source]:18s} {len(v):5d} objects, {(~whole).sum():4d} fragment-like (< {LOW}); "
                f"high-DNA (> {HIGH}): {high.sum():4d} = {entry['high_share_of_whole']:.1%} of whole objects")
        if source != "annotated":
            merge = g["status"].to_numpy() == "merge"
            covers = g["covers"].to_numpy()
            kept = ~merge
            entry.update(
                high_merges=int((high & merge).sum()),
                high_merges_of_two=int((high & merge & (covers == 2)).sum()),
                high_merges_of_three_or_more=int((high & merge & (covers >= 3)).sum()),
                high_share_without_merges=float((high & kept).sum() / max((whole & kept).sum(), 1)),
            )
            line += (f"\n  {'':18s} of which {entry['high_merges']} are merges ({entry['high_merges_of_two']} of two "
                     f"annotated nuclei, {entry['high_merges_of_three_or_more']} of three or more); "
                     f"without merges the share is {entry['high_share_without_merges']:.1%}")
        summary["dna_profile"][source] = entry
        print(line)

    values = {src: g["dna_relative"].to_numpy() for src, g in objects.groupby("source", sort=False)}
    plots.step_histograms(values, np.arange(0, 4.01, 0.05),
                          "DNA content, relative to the image's median annotated nucleus (last bin includes ≥ 4)",
                          "DNA content from each set of outlines, test split",
                          args.out / "dna_content.png", markers={1.0: "median", 2.0: "2× median"})
    errors = {m: g["dna_error"].to_numpy() for m, g in both.groupby("method", sort=False)}
    plots.step_histograms(errors, np.arange(-0.5, 0.501, 0.02), "DNA content error vs annotated outline",
                          "Per-nucleus DNA content error, nuclei both methods found",
                          args.out / "dna_error.png", markers={0.0: "no error"})
    plots.count_scatter(counts, args.out / "counts.png")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved tables, summary.json and figures to {args.out}")


if __name__ == "__main__":
    main()
