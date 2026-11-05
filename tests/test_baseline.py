"""Behaviour of the baseline's building blocks on shapes with known answers."""

import numpy as np
from scipy import ndimage as ndi
from skimage.measure import label
from skimage.morphology import h_maxima

from nuclei.baseline import (BaselineParams, deep_peaks, seeds_shape, seeds_shape_relative,
                             seeds_shape_v1, segment)


def two_touching_disks():
    # Radius 12, centres 20 px apart: a figure-of-eight with a neck 6.6 px
    # half-wide, i.e. 45% lower than the 12 px peaks.
    yy, xx = np.mgrid[:60, :80]
    return ((yy - 30) ** 2 + (xx - 28) ** 2 < 144) | ((yy - 30) ** 2 + (xx - 48) ** 2 < 144)


def waisted_nucleus(waist):
    # One elongated nucleus (70 x 22 px) pinched in the middle by `waist` px.
    yy, xx = np.mgrid[:60, :100]
    x, y = xx - 50, yy - 30
    half_width = 11 * np.sqrt(np.clip(1 - (x / 35) ** 2, 0, None)) - waist * np.exp(-x ** 2 / 128)
    return np.abs(y) < half_width


def as_image(mask, seed=0):
    rng = np.random.default_rng(seed)
    return (100 + 1000 * mask + rng.normal(0, 10, mask.shape)).astype(np.uint16)


def count(mask, **kw):
    return segment(as_image(mask), BaselineParams(sigma=1.0, threshold="otsu", **kw)).max()


def test_equal_peaks_with_a_shallow_dip_get_one_seed():
    # Two peaks of exactly equal height (5) with a dip to 4, depth 3.
    # scikit-image's h_maxima keeps both; deep_peaks merges them.
    f = np.array([[0, 5, 4, 5, 0]], dtype=float)
    assert label(h_maxima(f, 3.0)).max() == 2
    assert deep_peaks(f, 3.0, f > 0).max() == 1


def test_a_deep_dip_still_gives_two_seeds():
    f = np.array([[0, 5, 1, 5, 0]], dtype=float)
    assert deep_peaks(f, 3.0, f > 0).max() == 2


def test_touching_disks_are_split_by_every_seed_strategy():
    m = two_touching_disks()
    assert count(m, seeds="shape", seed_value=2.0) == 2
    assert count(m, seeds="shape_rel", seed_value=0.3) == 2
    assert count(m, seeds="intensity", seed_value=4.0) == 2


def test_slightly_waisted_nucleus_stays_whole():
    # Distance-transform ridge: ~9.9 px peaks, 9.0 px at the waist (a 9% dip).
    # The shape is mirror-symmetric, so its two peaks are exactly equal. In
    # real masks equal peaks are common too: distance values are square roots
    # of whole numbers.
    m = waisted_nucleus(2.0)
    dist = ndi.distance_transform_edt(m)
    assert seeds_shape(dist, 1.0, m).max() == 1
    assert seeds_shape_relative(dist, 0.1, m).max() == 1
    # Version 1 cut it in two whatever h was.
    assert seeds_shape_v1(dist, 4.0).max() == 2


def test_relative_rule_depends_on_the_fraction():
    # Waist of 3 px: peaks 10 px, neck 8 px, a 20% dip.
    m = waisted_nucleus(3.0)
    dist = ndi.distance_transform_edt(m)
    assert seeds_shape_relative(dist, 0.1, m).max() == 2
    assert seeds_shape_relative(dist, 0.3, m).max() == 1


def test_empty_field_returns_nothing():
    rng = np.random.default_rng(0)
    noise = (150 + rng.normal(0, 10, (100, 120))).astype(np.uint16)
    assert segment(noise, BaselineParams(threshold="li")).max() == 0
    # With the guard switched off, a global threshold segments the noise.
    assert segment(noise, BaselineParams(threshold="li", min_contrast=0.0)).max() > 0
