"""Unit tests for geometry and evaluation primitives."""
from __future__ import annotations

from helmet_safety.eval.metrics import aggregate, greedy_match, prf
from helmet_safety.schemas import BBox


def test_iou_identical_and_disjoint():
    a = BBox(0, 0, 10, 10)
    assert a.iou(BBox(0, 0, 10, 10)) == 1.0
    assert a.iou(BBox(20, 20, 30, 30)) == 0.0
    assert abs(a.iou(BBox(5, 0, 15, 10)) - 1 / 3) < 1e-9


def test_clip_keeps_box_inside_image():
    b = BBox(-5, -5, 100, 100).clip(50, 50)
    assert (b.x1, b.y1, b.x2, b.y2) == (0, 0, 50, 50)


def test_greedy_match_perfect():
    gt = [(0, BBox(0, 0, 10, 10)), (1, BBox(20, 20, 30, 30))]
    pred = [(0, BBox(0, 0, 10, 10), 0.9), (1, BBox(20, 20, 30, 30), 0.8)]
    assert greedy_match(pred, gt) == (2, 0, 0)


def test_greedy_match_counts_fp_fn_and_respects_class():
    gt = [(0, BBox(0, 0, 10, 10))]
    pred = [(1, BBox(0, 0, 10, 10), 0.9),  # wrong class -> FP
            (0, BBox(100, 100, 110, 110), 0.7)]  # far away -> FP
    tp, fp, fn = greedy_match(pred, gt)
    assert (tp, fp, fn) == (0, 2, 1)


def test_greedy_match_one_gt_consumed_once():
    gt = [(0, BBox(0, 0, 10, 10))]
    pred = [(0, BBox(0, 0, 10, 10), 0.9), (0, BBox(1, 1, 11, 11), 0.8)]
    assert greedy_match(pred, gt) == (1, 1, 0)


def test_prf_and_aggregate():
    assert prf(1, 1, 1)["f1"] == 0.5
    r = aggregate([{"tp": 2, "fp": 0, "fn": 0}, {"tp": 0, "fp": 1, "fn": 1}])
    assert r["tp"] == 2 and r["fp"] == 1 and r["fn"] == 1
    assert r["precision"] == round(2 / 3, 4)
