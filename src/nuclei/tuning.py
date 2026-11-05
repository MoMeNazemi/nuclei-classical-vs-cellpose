"""Run many baseline configurations over many images, using several CPU cores.

Each image is independent of the others, so images are handed out to worker
processes and their result rows are collected at the end. On Windows this
only works when the calling script protects its entry point with
`if __name__ == "__main__":`, which all scripts in this project do.
"""

from __future__ import annotations

import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

import pandas as pd
from scipy import ndimage as ndi

from nuclei.baseline import BaselineParams, foreground, smooth_image, split_nuclei
from nuclei.evaluate import evaluate
from nuclei.io import Sample, load_image, load_mask


def default_workers() -> int:
    """Leave one core free so the laptop stays usable; more than 8 rarely helps here."""
    return max(1, min(8, (os.cpu_count() or 2) - 1))


def evaluate_configs_on_image(sample: Sample, configs: list[BaselineParams],
                              with_objects: bool = False):
    """Run every configuration on one image and score it.

    Steps shared by several configurations (smoothing, thresholding,
    distance transform, blurring) are computed once per image and reused.
    Returns (rows, objects): one score row per configuration, and, if asked,
    one table of true nuclei per configuration.
    """
    img, gt = load_image(sample.image_path), load_mask(sample.mask_path)
    smoothed, masks, dists, blurs = {}, {}, {}, {}
    rows, objects = [], []
    for p in configs:
        if p.sigma not in smoothed:
            smoothed[p.sigma] = smooth_image(img, p.sigma)
        key = (p.sigma, p.threshold, p.factor, p.min_contrast)
        if key not in masks:
            masks[key] = foreground(smoothed[p.sigma], p.threshold, p.factor, p.min_contrast)
        fg = masks[key]
        dist = blurred = None
        if p.seeds == "intensity":
            if p.seed_value not in blurs:
                blurs[p.seed_value] = smooth_image(img, p.seed_value)
            blurred = blurs[p.seed_value]
        else:
            if key not in dists:
                dists[key] = ndi.distance_transform_edt(fg)
            dist = dists[key]

        ev = evaluate(gt, split_nuclei(fg, img, p, dist=dist, blurred=blurred))
        rows.append(dict(name=sample.name, **p.as_dict(), **ev.counts()))
        if with_objects:
            objects.append(ev.objects().assign(name=sample.name, **p.as_dict()))
    return rows, objects


def run(samples: list[Sample], configs: list[BaselineParams], workers: int,
        with_objects: bool = False) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Score every configuration on every image. Returns (scores, objects)."""
    job = partial(evaluate_configs_on_image, configs=configs, with_objects=with_objects)
    rows, objects = [], []
    start = time.time()

    def collect(result, done):
        rows.extend(result[0])
        objects.extend(result[1])
        if done % 10 == 0 or done == len(samples):
            elapsed = time.time() - start
            print(f"  {done}/{len(samples)} images, {elapsed:.0f} s elapsed, "
                  f"~{elapsed / done * (len(samples) - done):.0f} s left", flush=True)

    if workers <= 1:
        for done, s in enumerate(samples, 1):
            collect(job(s), done)
    else:
        # "spawn" starts each worker fresh, as Windows always does, so the code
        # behaves the same on every operating system.
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            futures = [pool.submit(job, s) for s in samples]
            for done, fut in enumerate(as_completed(futures), 1):
                collect(fut.result(), done)

    scores = pd.DataFrame(rows)
    return scores, (pd.concat(objects, ignore_index=True) if objects else None)
