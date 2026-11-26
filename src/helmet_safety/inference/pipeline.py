"""End-to-end inference pipeline on real frames.

    frame -> [tiling?] YOLO -> [temporal vote?] -> alert policy -> sinks

All three switches are independent so an A/B experiment can change exactly
one of them with everything else frozen.
"""
from __future__ import annotations

import time
from typing import Any, Dict

import numpy as np

from ..alert.policy import AlertManager, LogSink, PolicyConfig
from ..config import InferenceConfig
from ..logging import get_logger
from ..schemas import EvidenceLevel, FrameResult
from .detector import YoloDetector
from .temporal import TemporalVoter
from .tiling import tiled_predict

log = get_logger(__name__)


class SafetyPipeline:
    def __init__(self, detector: YoloDetector, cfg: InferenceConfig,
                 alert_cfg: PolicyConfig | None = None, sinks=None):
        self.detector = detector
        self.cfg = cfg
        self.voter = TemporalVoter(cfg.temporal_window, cfg.temporal_min_votes,
                                   cfg.temporal_iou)
        acfg = alert_cfg or PolicyConfig()
        if sinks is None:
            sinks = [LogSink()]
        self.alerts = AlertManager(acfg, sinks)
        self.frame_id = 0

    def reset(self) -> None:
        self.frame_id = 0
        self.voter.reset()
        self.alerts.reset()

    def process(self, frame: np.ndarray, emit_alerts: bool = True) -> FrameResult:
        t0 = time.perf_counter()
        self.frame_id += 1
        if self.cfg.enable_tiling:
            dets = tiled_predict(self.detector, frame, self.cfg.tile_size,
                                 self.cfg.tile_overlap, self.cfg.iou)
        else:
            dets = self.detector.predict(frame, source="full")

        if self.cfg.enable_temporal:
            confirmed = self.voter.update(dets)
        else:
            confirmed = list(dets)

        events = self.alerts.emit(confirmed) if emit_alerts else []
        return FrameResult(
            frame_id=self.frame_id,
            detections=dets,
            confirmed=confirmed,
            alerts=events,
            latency_ms=(time.perf_counter() - t0) * 1000,
            source="real-image",
            evidence_level=EvidenceLevel.MEASURED,
        )

    def describe(self) -> Dict[str, Any]:
        return {
            "detector": self.detector.describe(),
            "tiling": {"enabled": self.cfg.enable_tiling, "tile_size": self.cfg.tile_size,
                       "overlap": self.cfg.tile_overlap},
            "temporal": {"enabled": self.cfg.enable_temporal,
                         "window": self.cfg.temporal_window,
                         "min_votes": self.cfg.temporal_min_votes},
            "alert": {"cooldown_frames": self.alerts.cfg.cooldown_frames,
                      "dedup": self.alerts.cfg.enable_dedup},
            "evidence_level": EvidenceLevel.MEASURED.value,
        }
