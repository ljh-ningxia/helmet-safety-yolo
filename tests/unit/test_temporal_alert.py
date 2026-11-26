"""Unit tests for temporal voting and alert policy."""
from __future__ import annotations

from helmet_safety.alert.policy import AlertManager, PolicyConfig
from helmet_safety.inference.temporal import TemporalVoter
from helmet_safety.schemas import AlertLevel, BBox, Detection


def _d(cls=1, conf=0.9):
    return Detection(BBox(10, 10, 30, 30), conf, cls,
                     "no-hardhat" if cls == 1 else "hardhat")


def test_temporal_requires_min_votes():
    v = TemporalVoter(window=5, min_votes=3, iou=0.4)
    assert v.update([_d()]) == []
    assert v.update([_d()]) == []
    out = v.update([_d()])
    assert len(out) == 1 and out[0].track_id == 1


def test_temporal_survives_a_single_missed_frame():
    v = TemporalVoter(window=5, min_votes=3, iou=0.4)
    for _ in range(3):
        v.update([_d()])
    # Window is [1,1,1]; a few missed frames keep 3 votes, so the track still
    # emits — that is exactly the flicker suppression we want.
    assert len(v.update([])) == 1
    assert len(v.update([])) == 1
    # Once the old votes are pushed out of the 5-frame window, it goes quiet.
    assert len(v.update([])) == 0


def test_temporal_validation():
    try:
        TemporalVoter(window=3, min_votes=5)
    except ValueError:
        pass
    else:
        raise AssertionError("min_votes > window must be rejected")


def test_alert_levels():
    m = AlertManager(PolicyConfig(enable_dedup=False))
    assert m.emit([_d()])[0].level == AlertLevel.L1
    assert m.emit([_d(), _d()])[0].level == AlertLevel.L2
    assert m.emit([_d()] * 4)[0].level == AlertLevel.L3


def test_alert_dedup_respects_cooldown():
    m = AlertManager(PolicyConfig(enable_dedup=True, cooldown_frames=3))
    assert len(m.emit([_d()])) == 1
    assert len(m.emit([_d()])) == 0
    assert len(m.emit([_d()])) == 0
    assert len(m.emit([_d()])) == 1


def test_no_alert_when_only_helmets():
    m = AlertManager(PolicyConfig(enable_dedup=False))
    assert m.emit([_d(0), _d(0)]) == []
