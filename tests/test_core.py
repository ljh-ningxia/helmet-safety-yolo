"""核心逻辑单测：重点验证「公平 A/B 约束」与「无真值保护」不被破坏。

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src import metrics                                  # noqa: E402
from src.pipeline import PipelineConfig                   # noqa: E402
from src.schema import Detection                         # noqa: E402
from src.synthetic import SyntheticSite                  # noqa: E402
from src.temporal import TemporalVoter                   # noqa: E402
from src.tiling import _slices, _nms                     # noqa: E402


def d(x1, y1, x2, y2, cls=0, conf=0.9):
    return Detection(x1, y1, x2, y2, conf, cls,
                     "helmet" if cls == 0 else "no_helmet")


class TestMetrics(unittest.TestCase):
    def test_iou(self):
        self.assertAlmostEqual(metrics.iou([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)
        self.assertAlmostEqual(metrics.iou([0, 0, 10, 10], [20, 20, 30, 30]), 0.0)

    def test_no_ground_truth_blocks_accuracy_claims(self):
        r = metrics.evaluate([[d(0, 0, 10, 10)]], None, [10.0])
        self.assertTrue(r["insufficient_ground_truth"])
        self.assertIsNone(r["precision"])
        self.assertIsNone(r["recall"])
        self.assertIsNone(r["map50"])

    def test_with_ground_truth(self):
        r = metrics.evaluate([[d(0, 0, 10, 10, cls=1)]],
                             [[d(0, 0, 10, 10, cls=1)]], [10.0])
        self.assertTrue(r["ground_truth"])
        self.assertEqual(r["per_class"][1]["tp"], 1)
        self.assertEqual(r["per_class"][1]["fp"], 0)

    def test_detection_count_is_not_accuracy(self):
        a = metrics.evaluate([[d(0, 0, 10, 10)] * 1], None, [10.0])
        b = metrics.evaluate([[d(0, 0, 10, 10)] * 5], None, [10.0])
        c = metrics.compare(a, b)
        self.assertEqual(c["total_detections"]["delta"], 4)
        self.assertIsNone(c["precision"]["delta"])


class TestFairAB(unittest.TestCase):
    def test_variant_changes_exactly_one_switch(self):
        base = PipelineConfig()
        for name, key in [("B_tiling", "enable_tiling"),
                          ("B_temporal", "enable_temporal"),
                          ("B_dedup", "enable_alert_dedup")]:
            v = base.variant(name)
            on = [k for k in ("enable_tiling", "enable_temporal",
                              "enable_alert_dedup") if getattr(v, k)]
            self.assertEqual(on, [key], f"{name} 引入了多项变动")
        self.assertEqual([k for k in ("enable_tiling", "enable_temporal",
                                      "enable_alert_dedup")
                          if getattr(base.variant("A"), k)], [])

    def test_other_conditions_frozen(self):
        a, b = PipelineConfig().variant("A"), PipelineConfig().variant("B_tiling")
        for k in ("conf", "iou", "imgsz", "tile_size", "tile_overlap",
                  "tile_scale", "hit_k", "cooldown_sec"):
            self.assertEqual(getattr(a, k), getattr(b, k), k)


class TestModules(unittest.TestCase):
    def test_slices_cover_whole_image(self):
        boxes = _slices(1280, 720, 640, 0.25)
        self.assertTrue(boxes)
        self.assertTrue(any(x2 == 1280 for _, _, x2, _ in boxes))
        self.assertTrue(any(y2 == 720 for _, _, _, y2 in boxes))

    def test_nms_removes_duplicates(self):
        kept = _nms([d(0, 0, 10, 10, conf=0.9), d(1, 1, 11, 11, conf=0.8)], 0.45)
        self.assertEqual(len(kept), 1)

    def test_temporal_needs_k_consecutive_frames(self):
        voter = TemporalVoter(hit_k=3)
        box = d(0, 0, 10, 10, cls=1)
        self.assertEqual(len(voter.update([box])), 0)
        self.assertEqual(len(voter.update([box])), 0)
        self.assertEqual(len(voter.update([box])), 1)

    def test_synthetic_is_deterministic(self):
        s = SyntheticSite(seed=7)
        a, ga = s.frame(3)
        b, gb = s.frame(3)
        self.assertTrue((a == b).all())
        self.assertEqual([x.to_dict() for x in ga], [x.to_dict() for x in gb])


if __name__ == "__main__":
    unittest.main(verbosity=2)
