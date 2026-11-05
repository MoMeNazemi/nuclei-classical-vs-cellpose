"""Checks of the Cellpose wrapper that need no model weights.

Skipped automatically when Cellpose isn't installed.
"""

import numpy as np
import pytest

pytest.importorskip("cellpose")

from cellpose import dynamics  # noqa: E402

from nuclei.evaluate import evaluate  # noqa: E402
from nuclei.pretrained import masks_from_output  # noqa: E402


def test_masks_rebuilt_from_ideal_network_output():
    # Fake a perfect network: flows computed from known nuclei (as Cellpose
    # does when preparing training data) and a cell probability that is
    # clearly positive inside nuclei. Stage 2 must give the nuclei back.
    labels = np.zeros((80, 100), dtype=np.int32)
    yy, xx = np.mgrid[:80, :100]
    labels[(yy - 25) ** 2 + (xx - 25) ** 2 < 144] = 1
    labels[(yy - 25) ** 2 + (xx - 50) ** 2 < 144] = 2   # touches nucleus 1
    labels[(yy - 60) ** 2 / 64 + (xx - 70) ** 2 / 400 < 1] = 3
    flows, _ = dynamics.masks_to_flows_gpu(labels)
    cellprob = np.where(labels > 0, 5.0, -5.0)
    pred = masks_from_output(flows * 5.0, cellprob)
    c = evaluate(labels, pred).counts()
    assert c["n_pred"] == 3 and c["tp_50"] == 3
    assert c["af1"] > 0.9


def test_higher_cellprob_threshold_removes_weak_objects():
    labels = np.zeros((60, 60), dtype=np.int32)
    yy, xx = np.mgrid[:60, :60]
    labels[(yy - 30) ** 2 + (xx - 30) ** 2 < 100] = 1
    flows, _ = dynamics.masks_to_flows_gpu(labels)
    cellprob = np.where(labels > 0, 0.5, -5.0)  # weakly positive nucleus
    assert masks_from_output(flows * 5.0, cellprob, cellprob_threshold=0.0).max() == 1
    assert masks_from_output(flows * 5.0, cellprob, cellprob_threshold=1.0).max() == 0
