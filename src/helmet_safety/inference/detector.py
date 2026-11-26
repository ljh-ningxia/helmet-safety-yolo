"""Real YOLO inference backend.

There is intentionally NO synthetic/demo detector in v2. Every number the
system reports must come from an actual trained weight file on real pixels.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from ..config import ROOT
from ..logging import get_logger
from ..schemas import BBox, Detection, EvidenceLevel

log = get_logger(__name__)


class DetectorError(RuntimeError):
    pass


class YoloDetector:
    """Thin, testable wrapper around Ultralytics YOLO.

    Lazy-loads the weights so unit tests and CLI ``--help`` do not require a
    model file. ``warmup`` is explicit to keep latency benchmarks honest.
    """

    def __init__(self, weights: str, classes: tuple = ("hardhat", "no-hardhat"),
                 conf: float = 0.35, iou: float = 0.5, imgsz: int = 640,
                 device: str = "cpu"):
        self.weights_path = Path(weights)
        if not self.weights_path.is_absolute():
            self.weights_path = ROOT / self.weights_path
        self.classes = tuple(classes)
        self.conf = float(conf)
        self.iou = float(iou)
        self.imgsz = int(imgsz)
        self.device = device
        self._model = None
        self._load_ms = 0.0

    @property
    def model(self):
        if self._model is None:
            if not self.weights_path.is_file():
                raise DetectorError(
                    f"weights not found: {self.weights_path}. "
                    "Run scripts/03_train.py first. v2 has no demo fallback.")
            from ultralytics import YOLO
            t0 = time.perf_counter()
            self._model = YOLO(str(self.weights_path))
            self._load_ms = (time.perf_counter() - t0) * 1000
            log.info("model loaded", weights=str(self.weights_path), load_ms=round(self._load_ms, 1))
        return self._model

    def warmup(self, shape=(640, 640, 3), n: int = 3) -> None:
        dummy = np.zeros(shape, dtype=np.uint8)
        for _ in range(n):
            self.model(dummy, imgsz=self.imgsz, conf=self.conf, iou=self.iou,
                       device=self.device, verbose=False)

    def predict(self, frame: np.ndarray, offset=(0, 0), source: str = "yolo") -> List[Detection]:
        if frame is None or frame.size == 0:
            raise DetectorError("empty frame")
        res = self.model(frame, imgsz=self.imgsz, conf=self.conf, iou=self.iou,
                         device=self.device, verbose=False)[0]
        out: List[Detection] = []
        if getattr(res, "boxes", None) is None:
            return out
        ox, oy = offset
        for xyxy, cls, cf in zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist(),
                                 res.boxes.conf.tolist()):
            c = int(cls)
            out.append(Detection(
                bbox=BBox(float(xyxy[0]) + ox, float(xyxy[1]) + oy,
                          float(xyxy[2]) + ox, float(xyxy[3]) + oy),
                conf=float(cf), cls=c,
                label=self.classes[c] if c < len(self.classes) else f"class_{c}",
                source=source,
            ))
        return out

    def describe(self) -> Dict[str, Any]:
        return {
            "backend": "ultralytics-yolo",
            "weights": str(self.weights_path.resolve()).replace("\\", "/"),
            "weights_exists": self.weights_path.is_file(),
            "classes": list(self.classes),
            "conf": self.conf, "iou": self.iou, "imgsz": self.imgsz,
            "device": self.device,
            "evidence_level": EvidenceLevel.MEASURED.value,
        }
