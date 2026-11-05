"""Measurements on images where the right numbers are known."""

import numpy as np
import pandas as pd
import pytest

from nuclei.measure import area_and_dna, background_level, found_by_all, image_unit


def test_area_and_dna_content():
    labels = np.zeros((20, 30), dtype=np.int32)
    labels[2:6, 2:6] = 1      # 16 px
    labels[10:15, 10:20] = 2  # 50 px
    signal = np.where(labels == 1, 3.0, 0.0) + np.where(labels == 2, 2.0, 0.0)
    area, dna = area_and_dna(labels, signal)
    assert list(area) == [16, 50]
    assert list(dna) == [48.0, 100.0]


def test_background_ignores_nuclei_and_their_rim():
    img = np.full((60, 60), 100.0)
    annotated = np.zeros((60, 60), dtype=np.int32)
    annotated[20:40, 20:40] = 1
    img[18:42, 18:42] = 900.0  # nucleus plus a 2 px blurred rim
    assert background_level(img, annotated) == pytest.approx(100.0)


def test_image_unit_ignores_micronuclei():
    area = np.array([10, 12, 400, 420, 410])
    dna = np.array([5.0, 6.0, 1000.0, 1100.0, 1050.0])
    assert image_unit(area, dna) == pytest.approx(1050.0)


def test_found_by_all_keeps_only_shared_nuclei():
    pairs = pd.DataFrame({
        "method": ["a", "a", "a", "b", "b"],
        "name":   ["img1", "img1", "img2", "img1", "img2"],
        "gt":     [0, 1, 0, 1, 5],
        "dna_error": [0.1, 0.2, 0.3, 0.4, 0.5],
    })
    both = found_by_all(pairs)
    # Only nucleus 1 of img1 was found by both methods.
    assert sorted(zip(both["method"], both["name"], both["gt"])) == [("a", "img1", 1), ("b", "img1", 1)]
