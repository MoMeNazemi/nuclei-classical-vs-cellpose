"""Per-nucleus measurements of the kind a screening biologist reads out.

DNA content: Hoechst binds DNA, so the background-subtracted intensity summed
over a nucleus (its integrated intensity) is proportional to how much DNA it
holds, as far as a 2D widefield image allows.

The values are put on a RELATIVE scale: 1 = the median annotated nucleus of
the same image. In a growing, untreated culture most cells are in G1, so the
median would sit near G1 and G2/M cells near 2. BBBC039 shows a different
compound in every image, though, and a compound that arrests cells can move
the median. So the scale is used to compare outlines with each other, not to
assign cell-cycle phases.

Background and scale both come from the annotated outlines, so every method
is measured against the same reference. A pipeline without annotations would
have to estimate them another way.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

BACKGROUND_MARGIN = 5  # px kept clear of annotated nuclei when estimating background
WHOLE_NUCLEUS_MIN_AREA = 25  # px; smaller objects (micronuclei) don't set the per-image scale


def background_level(img: np.ndarray, annotated: np.ndarray) -> float:
    """Median intensity of pixels at least BACKGROUND_MARGIN px from any annotated nucleus.

    Taken from the annotation, not from either method, so the background
    is identical for every method being compared.
    """
    near = ndi.binary_dilation(annotated > 0, iterations=BACKGROUND_MARGIN)
    return float(np.median(img[~near]))


def area_and_dna(labels: np.ndarray, signal: np.ndarray):
    """Area (px) and DNA content (summed signal) of objects 1..N of a label image."""
    n = int(labels.max())
    area = np.bincount(labels.ravel(), minlength=n + 1)[1:]
    dna = np.bincount(labels.ravel(), weights=signal.ravel(), minlength=n + 1)[1:]
    return area, dna


def image_unit(area: np.ndarray, dna: np.ndarray) -> float:
    """Median DNA content of the annotated whole nuclei in one image.

    Dividing by it puts every image on the same relative scale (1 = the
    image's median nucleus). Micronuclei are left out so they can't drag
    the median down.
    """
    whole = dna[area >= WHOLE_NUCLEUS_MIN_AREA]
    return float(np.median(whole)) if len(whole) else np.nan


def found_by_all(pairs: pd.DataFrame) -> pd.DataFrame:
    """Rows of `pairs` for the annotated nuclei that EVERY method found.

    `pairs` has one row per (method, image `name`, annotated nucleus `gt`)
    match. Comparing methods on their own matches mixes two things: a method
    that skips difficult nuclei can look accurate on the easy ones it keeps.
    Restricting to nuclei all methods found compares like with like.
    """
    methods = pairs["method"].nunique()
    found_by = pairs.groupby(["name", "gt"])["method"].transform("nunique")
    return pairs[found_by == methods]
