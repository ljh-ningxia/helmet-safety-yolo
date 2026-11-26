"""Transparent evaluation primitives.

These are deliberately simple, dependency-free implementations so that every
number in an experiment report can be traced to a readable function instead
of a library black box. Official mAP is still taken from Ultralytics validation;
these helpers power slice analysis and single-variable A/B comparisons.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from ..schemas import BBox


def greedy_match(pred: Sequence[Tuple[int, BBox, float]],
                 gt: Sequence[Tuple[int, BBox]],
                 iou_thr: float = 0.5) -> Tuple[int, int, int]:
    """Class-aware greedy IoU matching -> (TP, FP, FN).

    Predictions are considered in descending confidence. A ground-truth box
    can be consumed at most once. This is NOT COCO mAP; it is an interpretable
    fixed-threshold diagnostic used for slices and A/B comparisons.
    """
    used: set = set()
    tp = 0
    for c, box, _conf in sorted(pred, key=lambda x: -x[2]):
        best_i, best_iou = -1, 0.0
        for i, (gc, gb) in enumerate(gt):
            if i in used or gc != c:
                continue
            v = box.iou(gb)
            if v > best_iou:
                best_i, best_iou = i, v
        if best_i >= 0 and best_iou >= iou_thr:
            used.add(best_i)
            tp += 1
    return tp, len(pred) - tp, len(gt) - len(used)


def prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-12)
    return {"precision": round(p, 4), "recall": round(r, 4),
            "f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn}


def aggregate(records: List[Dict[str, int]]) -> Dict[str, float]:
    tp = sum(r.get("tp", 0) for r in records)
    fp = sum(r.get("fp", 0) for r in records)
    fn = sum(r.get("fn", 0) for r in records)
    return prf(tp, fp, fn)


def load_yolo_gt(txt_path, w: int, h: int) -> List[Tuple[int, BBox]]:
    from pathlib import Path
    p = Path(txt_path)
    out = []
    if not p.exists():
        return out
    for ln in p.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        c, cx, cy, bw, bh = (float(v) for v in ln.split()[:5])
        out.append((int(c), BBox((cx - bw / 2) * w, (cy - bh / 2) * h,
                                 (cx + bw / 2) * w, (cy + bh / 2) * h)))
    return out
