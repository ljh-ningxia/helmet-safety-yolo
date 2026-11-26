"""Unit tests for tiled inference and NMS."""
from __future__ import annotations

import numpy as np

from helmet_safety.inference.tiling import make_tiles, nms, tiled_predict
from helmet_safety.schemas import BBox, Detection


def test_tiles_cover_whole_image():
    tiles = make_tiles(640, 640, 320, 0.25)
    assert tiles
    assert min(t[0] for t in tiles) == 0 and min(t[1] for t in tiles) == 0
    assert max(t[2] for t in tiles) == 640 and max(t[3] for t in tiles) == 640
    assert tiles[-1] == (320, 320, 640, 640)


def test_tiles_on_small_image_is_one_full_tile():
    assert make_tiles(200, 200, 320, 0.25) == [(0, 0, 200, 200)]


def test_nms_suppresses_overlap_but_keeps_other_class():
    ds = [Detection(BBox(0, 0, 10, 10), 0.9, 0, "hardhat"),
          Detection(BBox(1, 1, 11, 11), 0.8, 0, "hardhat"),
          Detection(BBox(1, 1, 11, 11), 0.7, 1, "no-hardhat")]
    out = nms(ds, 0.5)
    assert len(out) == 2
    assert out[0].conf == 0.9


class _FakeDetector:
    """Returns one box at the centre of whatever crop it receives."""

    def __init__(self):
        self.calls = 0

    def predict(self, frame, offset=(0, 0), source="full"):
        self.calls += 1
        h, w = frame.shape[:2]
        return [Detection(BBox(w * 0.4 + offset[0], h * 0.4 + offset[1],
                               w * 0.6 + offset[0], h * 0.6 + offset[1]),
                          0.9, 1, "no-hardhat", source=source)]


def test_tiled_predict_uses_multiple_crops_and_merges():
    d = _FakeDetector()
    out = tiled_predict(d, np.zeros((640, 640, 3), np.uint8), 320, 0.25, 0.5)
    assert d.calls == len(make_tiles(640, 640, 320, 0.25))
    assert len(out) >= 1
    assert all(0 <= x.bbox.x1 <= 640 for x in out)
