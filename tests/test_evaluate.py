"""Check the ruler before measuring anything with it.

Each test builds a tiny case where the right answer can be worked out by
hand, then checks that the evaluation code gets it. Run with:  pytest
"""

import numpy as np
import pandas as pd
import pytest

from nuclei.evaluate import IOU_THRESHOLDS, evaluate, summarize
from nuclei.io import decode_mask, decode_mask_first_channel


def blank(shape=(40, 60)):
    return np.zeros(shape, dtype=np.int32)


def test_thresholds_match_the_paper():
    assert list(IOU_THRESHOLDS) == [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]


def test_perfect_prediction():
    gt = blank()
    gt[5:15, 5:15] = 1
    gt[20:30, 30:45] = 2
    c = evaluate(gt, gt.copy()).counts()
    assert c["tp_50"] == 2 and c["fp_50"] == 0 and c["fn_50"] == 0
    assert c["af1"] == pytest.approx(1.0)
    assert c["merges"] == c["splits"] == c["missed"] == c["extra"] == 0


def test_label_numbers_do_not_matter():
    gt = blank()
    gt[5:15, 5:15] = 1
    pred = blank()
    pred[5:15, 5:15] = 77  # same object, different number
    assert evaluate(gt, pred).counts()["f1_50"] == pytest.approx(1.0)


def test_shifted_object():
    # 10x10 square shifted 2 px: overlap 80, union 120, IoU = 0.667.
    # Correct at thresholds 0.50-0.65 (4 of 9), wrong from 0.70 up.
    gt = blank()
    gt[5:15, 5:15] = 1
    pred = blank()
    pred[5:15, 7:17] = 1
    ev = evaluate(gt, pred)
    assert ev.iou[0, 0] == pytest.approx(80 / 120)
    assert list(ev.tp) == [1, 1, 1, 1, 0, 0, 0, 0, 0]
    assert ev.counts()["af1"] == pytest.approx(4 / 9)


def test_merge():
    # Two touching nuclei (100 px and 60 px) predicted as one 160 px object.
    # The object covers both, so it is a merge. Its IoU with the big nucleus
    # is still 100/160 = 0.625, so the big one counts as found.
    gt = blank()
    gt[5:15, 5:15] = 1
    gt[5:15, 15:21] = 2
    pred = blank()
    pred[5:15, 5:21] = 1
    ev = evaluate(gt, pred)
    c = ev.counts()
    assert c["merges"] == 1 and c["splits"] == 0
    assert c["tp_50"] == 1 and c["fn_50"] == 1 and c["fp_50"] == 0
    assert list(ev.gt_status) == ["matched", "merged"]
    assert list(ev.pred_status) == ["merge"]
    assert list(ev.pred_covers) == [2]


def test_merge_size_counts_every_nucleus_covered():
    # Three nuclei in a row predicted as one object, next to a correct one.
    gt = blank()
    gt[2:8, 2:8], gt[2:8, 8:14], gt[2:8, 14:20] = 1, 2, 3
    gt[20:26, 20:26] = 4
    pred = blank()
    pred[2:8, 2:20] = 1
    pred[20:26, 20:26] = 2
    ev = evaluate(gt, pred)
    assert list(ev.pred_covers) == [3, 1]
    assert list(ev.pred_status) == ["merge", "matched"]


def test_split_into_halves():
    # One 10x20 nucleus predicted as two halves: each half has IoU = 0.5,
    # which is NOT above 0.5, so nothing matches.
    gt = blank()
    gt[5:15, 5:25] = 1
    pred = blank()
    pred[5:15, 5:15] = 1
    pred[5:15, 15:25] = 2
    c = evaluate(gt, pred).counts()
    assert c["splits"] == 1 and c["merges"] == 0
    assert c["tp_50"] == 0 and c["fn_50"] == 1 and c["fp_50"] == 2


def test_thin_sliver_is_a_split_fragment_not_an_extra_object():
    # A 10x30 nucleus cut into two big pieces and a 2 px-wide sliver. The
    # sliver's IoU with the nucleus is only 20/300 = 0.07, so an IoU > 0.1
    # rule ignores it; by coverage it lies entirely inside the nucleus.
    gt = blank()
    gt[5:15, 5:35] = 1
    pred = blank()
    pred[5:15, 5:19] = 1
    pred[5:15, 19:21] = 2
    pred[5:15, 21:35] = 3
    ev = evaluate(gt, pred)
    assert list(ev.pred_status) == ["split", "split", "split"]
    assert ev.counts()["extra"] == 0
    assert ev.counts()["splits"] == 1


def test_neighbour_overreach_counts_as_missed_not_merged():
    # Nucleus A is predicted well but the prediction spills 3 columns into
    # its neighbour B, and B gets no prediction of its own. That is a missed
    # nucleus, not a merge (B is only 30% covered). The IoU > 0.1 rule calls
    # it a merge: IoU(B, prediction) = 30/200 = 0.15.
    gt = blank()
    gt[5:15, 5:15] = 1
    gt[5:15, 15:25] = 2
    pred = blank()
    pred[5:15, 5:18] = 1
    ev = evaluate(gt, pred)
    c = ev.counts()
    assert list(ev.gt_status) == ["matched", "missed"]
    assert c["merges"] == 0 and c["merges_ref"] == 1


def test_undersized_detection_is_a_poor_outline_not_a_miss():
    gt = blank()
    gt[5:25, 5:25] = 1        # 400 px nucleus
    pred = blank()
    pred[10:18, 10:18] = 1    # 64 px prediction inside it: IoU 0.16
    ev = evaluate(gt, pred)
    assert list(ev.gt_status) == ["poor_fit"]
    assert list(ev.pred_status) == ["poor_fit"]


def test_missed_and_extra():
    gt = blank()
    gt[5:15, 5:15] = 1
    pred = blank()
    pred[25:35, 40:50] = 1  # nowhere near the true nucleus
    c = evaluate(gt, pred).counts()
    assert c["missed"] == 1 and c["extra"] == 1
    assert c["f1_50"] == 0.0


def test_area_bias():
    # Prediction 1 px larger on every side: 12x12 = 144 px vs 100 px true.
    gt = blank()
    gt[5:15, 5:15] = 1
    pred = blank()
    pred[4:16, 4:16] = 1
    s = summarize(pd.DataFrame([evaluate(gt, pred).counts()]))
    assert s["area_bias"] == pytest.approx(0.44)


def test_empty_image_is_not_scored():
    c = evaluate(blank(), blank()).counts()
    assert c["n_gt"] == 0 and c["n_pred"] == 0
    assert np.isnan(c["f1_50"]) and np.isnan(c["af1"])


def test_decode_touching_nuclei_by_colour():
    # Two touching patches that differ only in green: full-colour decoding
    # separates them, first-channel decoding merges them.
    raw = np.zeros((10, 20, 3), dtype=np.uint8)
    raw[2:8, 2:10] = [200, 10, 10]
    raw[2:8, 10:18] = [200, 120, 10]
    assert decode_mask(raw).max() == 2
    assert decode_mask_first_channel(raw).max() == 1
