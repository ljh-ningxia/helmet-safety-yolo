"""Pipeline tests use a stub detector so CI never needs model weights."""
from __future__ import annotations

import numpy as np

from helmet_safety.alert.policy import PolicyConfig
from helmet_safety.config import load_config
from helmet_safety.inference.pipeline import SafetyPipeline
from helmet_safety.schemas import BBox, Detection, EvidenceLevel


class StubDetector:
    def __init__(self):
        self.calls = 0

    def predict(self, frame, offset=(0, 0), source="full"):
        self.calls += 1
        return [Detection(BBox(10, 10, 40, 40), 0.9, 1, "no-hardhat", source=source)]

    def describe(self):
        return {"backend": "stub", "evidence_level": "SELFTEST_ONLY"}


def test_pipeline_baseline_and_switches():
    cfg = load_config(overrides={
        "inference": {"enable_tiling": False, "enable_temporal": False,
                      "temporal_min_votes": 1}})
    p = SafetyPipeline(StubDetector(), cfg.inference, PolicyConfig(enable_dedup=False))
    r = p.process(np.zeros((320, 320, 3), np.uint8))
    assert r.evidence_level == EvidenceLevel.MEASURED
    assert len(r.detections) == 1
    assert r.confirmed and r.alerts
    assert r.latency_ms >= 0
    assert "detector" in p.describe()


def test_pipeline_temporal_switch_changes_output():
    cfg = load_config(overrides={
        "inference": {"enable_tiling": False, "enable_temporal": True,
                      "temporal_min_votes": 3}})
    p = SafetyPipeline(StubDetector(), cfg.inference, PolicyConfig(enable_dedup=False))
    r1 = p.process(np.zeros((320, 320, 3), np.uint8))
    assert r1.confirmed == []  # not enough votes yet
    for _ in range(2):
        r = p.process(np.zeros((320, 320, 3), np.uint8))
    assert len(r.confirmed) == 1
