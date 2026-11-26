"""Tiled (sliced) inference for small, distant helmets.

A 640px input compresses a far-away worker's head into a few pixels. Splitting
the frame into overlapping tiles restores apparent object scale at the cost of
extra forward passes. Class-agnostic NMS merges duplicate detections on the
overlap regions. The trade-off (recall vs latency vs false positives) is
measured, never assumed — see experiments/results/ab_*.json.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from ..schemas import Detection


def make_tiles(width: int, height: int, tile: int, overlap: float) -> List[Tuple[int, int, int, int]]:
    if tile <= 0:
        raise ValueError("tile size must be positive")
    step = max(1, int(tile * (1.0 - overlap)))
    xs = list(range(0, max(1, width - tile + 1), step))
    ys = list(range(0, max(1, height - tile + 1), step))
    if not xs or xs[-1] + tile < width:
        xs.append(max(0, width - tile))
    if not ys or ys[-1] + tile < height:
        ys.append(max(0, height - tile))
    return [(x, y, min(x + tile, width), min(y + tile, height))
            for y in ys for x in xs]


def nms(dets: Sequence[Detection], iou_thr: float = 0.5) -> List[Detection]:
    """Greedy class-aware NMS; assumes input sorted by confidence descending."""
    kept: List[Detection] = []
    for d in sorted(dets, key=lambda x: -x.conf):
        if any(k.cls == d.cls and d.bbox.iou(k.bbox) >= iou_thr for k in kept):
            continue
        kept.append(d)
    return kept


def tiled_predict(detector, frame: np.ndarray, tile: int, overlap: float,
                  nms_iou: float = 0.5) -> List[Detection]:
    h, w = frame.shape[:2]
    if w <= tile and h <= tile:
        return detector.predict(frame, source="full")
    boxes = make_tiles(w, h, tile, overlap)
    all_dets: List[Detection] = []
    for x1, y1, x2, y2 in boxes:
        crop = frame[y1:y2, x1:x2]
        all_dets.extend(detector.predict(crop, offset=(x1, y1), source="tile"))
    return nms(all_dets, nms_iou)
