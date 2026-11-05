"""Classical nuclei segmentation in two stages.

1. Foreground: smooth the image, apply one global threshold, fill holes.
   This decides WHICH pixels belong to nuclei.
2. Splitting: place one seed per nucleus, then grow the seeds with a
   watershed until they meet. This decides HOW the foreground is cut into
   individual nuclei.

Seed strategies (BaselineParams.seeds):

- "shape": peaks of the distance transform (each pixel's distance to the
  background). A round nucleus has one peak; two touching nuclei have two
  peaks with a lower neck between them. A peak becomes a seed if it rises at
  least h pixels above that neck (seed_value = h).
- "shape_rel": the same, but the neck must be lower by a FRACTION r of the
  peak height (seed_value = r). A fixed h treats a 1 px dip in a 3 px-wide
  fragment and in a 25 px-wide nucleus alike; a fraction scales with size.
- "intensity": brightness peaks after blurring at roughly nucleus scale
  (seed_value = blur sigma in px), and the watershed cuts along dark valleys.
  Similar to CellProfiler's intensity-based declumping.
- "shape_v1": the first version, kept only to reproduce its results. It used
  scikit-image's h_maxima, which keeps two peaks of EXACTLY equal height
  however shallow the dip between them. Distance-transform values are square
  roots of whole numbers, so such ties are common, and one nucleus with two
  equal bumps on its ridge got cut in two whatever h was.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.filters import gaussian, threshold_li, threshold_otsu
from skimage.measure import label
from skimage.morphology import h_maxima, local_maxima, reconstruction
from skimage.segmentation import relabel_sequential, watershed

THRESHOLD_METHODS = {"otsu": threshold_otsu, "li": threshold_li}
SEED_METHODS = ("shape", "shape_rel", "intensity", "shape_v1")
MIN_SEED_DISTANCE = 7  # px between intensity seeds, about a quarter of the median nucleus diameter (28 px)

# Empty-field guard. A global threshold always splits the histogram in two,
# even when the field is empty, and then segments noise. If what it calls
# foreground is less than this many times brighter than the background,
# there is nothing there and we return no nuclei.
MIN_CONTRAST = 1.5


@dataclass(frozen=True)
class BaselineParams:
    sigma: float = 1.0       # Gaussian smoothing before thresholding (px)
    threshold: str = "otsu"  # global threshold method, see THRESHOLD_METHODS
    factor: float = 1.0      # threshold correction factor: >1 gives tighter outlines, <1 looser
    seeds: str = "shape"     # how to place one seed per nucleus, see SEED_METHODS
    seed_value: float = 1.0  # h (px) for "shape", r (fraction) for "shape_rel", blur sigma (px) for "intensity"
    min_size: int = 16       # objects smaller than this are removed (px); 16 as in the authors' code
    min_contrast: float = MIN_CONTRAST  # empty-field guard; 0 switches it off

    def as_dict(self) -> dict:
        return asdict(self)


def smooth_image(img: np.ndarray, sigma: float) -> np.ndarray:
    return gaussian(img.astype(np.float32), sigma=sigma, preserve_range=True)


def foreground(smooth: np.ndarray, threshold: str, factor: float = 1.0,
               min_contrast: float = MIN_CONTRAST) -> np.ndarray:
    """Binary nucleus mask from an already smoothed image."""
    t = THRESHOLD_METHODS[threshold](smooth) * factor
    fg = ndi.binary_fill_holes(smooth > t)
    if min_contrast > 0 and fg.any() and not fg.all():
        contrast = np.median(smooth[fg]) / max(float(np.median(smooth[~fg])), 1e-9)
        if contrast < min_contrast:
            return np.zeros_like(fg)  # empty field: nothing is clearly brighter than the background
    return fg


def deep_peaks(f: np.ndarray, depth: float, mask: np.ndarray) -> np.ndarray:
    """One seed per peak of f that rises more than `depth` above the lowest
    point on the way to any other peak at least as high.

    Lower every value by `depth` and let the peaks "flood" downhill without
    rising above f (morphological reconstruction). Peaks joined by a pass
    shallower than `depth` merge into one flat top, so they share one seed,
    including peaks of exactly equal height.
    """
    flooded = reconstruction(f - depth, f, method="dilation")
    return label(local_maxima(flooded) & mask)


def seeds_shape(dist: np.ndarray, h: float, fg: np.ndarray) -> np.ndarray:
    return deep_peaks(dist, h, fg)


def seeds_shape_relative(dist: np.ndarray, r: float, fg: np.ndarray) -> np.ndarray:
    """Distance-transform peaks whose neck is lower by at least a fraction r.

    Trick: on a log scale, a fixed difference is a fixed ratio. A neck at
    height S below a peak at height H passes if S/H <= 1 - r, which is the
    same as log(H) - log(S) >= -log(1 - r). So the same peak test on
    log(distance) with that depth does the job.
    """
    log_dist = np.zeros_like(dist)
    inside = dist > 0  # inside the mask the distance is at least 1, so log >= 0
    log_dist[inside] = np.log(dist[inside])
    return deep_peaks(log_dist, -np.log1p(-r), fg)


def seeds_shape_v1(dist: np.ndarray, h: float) -> np.ndarray:
    return label(h_maxima(dist, h))


def seeds_intensity(blurred: np.ndarray, fg: np.ndarray) -> np.ndarray:
    coords = peak_local_max(blurred, min_distance=MIN_SEED_DISTANCE, labels=label(fg), exclude_border=False)
    markers = np.zeros(fg.shape, dtype=np.int32)
    markers[tuple(coords.T)] = np.arange(1, len(coords) + 1)
    return markers


def remove_small(labels: np.ndarray, min_size: int) -> np.ndarray:
    """Delete labelled objects smaller than min_size pixels.

    Written out by hand because scikit-image keeps changing the arguments of
    remove_small_objects between versions.
    """
    areas = np.bincount(labels.ravel())
    too_small = areas < min_size
    too_small[0] = False  # never touch background
    out = labels.copy()
    out[too_small[labels]] = 0
    return out


def split_nuclei(fg: np.ndarray, img: np.ndarray, p: BaselineParams,
                 dist: np.ndarray | None = None, blurred: np.ndarray | None = None) -> np.ndarray:
    """Cut a foreground mask into individual nuclei.

    `dist` and `blurred` can be passed in when they were already computed,
    which saves time when trying many parameter values on one image.
    """
    if p.seeds in ("shape", "shape_rel", "shape_v1"):
        if dist is None:
            dist = ndi.distance_transform_edt(fg)
        if p.seeds == "shape":
            markers = seeds_shape(dist, p.seed_value, fg)
        elif p.seeds == "shape_rel":
            markers = seeds_shape_relative(dist, p.seed_value, fg)
        else:
            markers = seeds_shape_v1(dist, p.seed_value)
        surface = -dist  # flood from the peaks, meet at the necks
    elif p.seeds == "intensity":
        if blurred is None:
            blurred = smooth_image(img, p.seed_value)
        markers = seeds_intensity(blurred, fg)
        surface = -blurred  # flood from the bright centres, meet in the dark valleys
    else:
        raise ValueError(f"Unknown seed method {p.seeds!r}; choose from {SEED_METHODS}")

    labels = watershed(surface, markers, mask=fg)

    # A blob that received no seed would silently vanish. Keep each as one object.
    orphan = fg & (labels == 0)
    if orphan.any():
        labels[orphan] = label(orphan)[orphan] + labels.max()

    labels = remove_small(labels, p.min_size)
    return relabel_sequential(labels)[0]


def segment(img: np.ndarray, p: BaselineParams) -> np.ndarray:
    """Full baseline: image in, label image out."""
    fg = foreground(smooth_image(img, p.sigma), p.threshold, p.factor, p.min_contrast)
    return split_nuclei(fg, img, p)


# The configuration chosen in the first version of step 2, reproduced exactly
# (no empty-field guard, scikit-image h_maxima), for before/after comparisons.
V1_PARAMS = BaselineParams(sigma=1.0, threshold="li", factor=1.0, seeds="shape_v1",
                           seed_value=1.0, min_size=16, min_contrast=0.0)
