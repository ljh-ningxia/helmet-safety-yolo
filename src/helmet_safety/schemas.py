"""Domain types shared across data, training, inference and serving layers."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class EvidenceLevel(str, Enum):
    """Provenance of a reported number. Never silently upgrade a level."""
    MEASURED = "MEASURED"        # produced by an actual run, with a log + config
    SELFTEST_ONLY = "SELFTEST_ONLY"  # v1 legacy marker; kept for reading old reports, never emitted by v2
    UNVERIFIED = "UNVERIFIED"    # quoted from a third party, not reproduced here


class AlertLevel(str, Enum):
    L1 = "L1"  # single person missing a hard hat
    L2 = "L2"  # 2-3 people, or repeated on the same track
    L3 = "L3"  # 4+ people, critical


@dataclass(frozen=True)
class BBox:
    """Absolute pixel coordinates, half-open [x1, y1, x2, y2)."""
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    def iou(self, other: "BBox") -> float:
        ix1, iy1 = max(self.x1, other.x1), max(self.y1, other.y1)
        ix2, iy2 = min(self.x2, other.x2), min(self.y2, other.y2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def clip(self, w: int, h: int) -> "BBox":
        return BBox(min(max(self.x1, 0.0), w), min(max(self.y1, 0.0), h),
                    min(max(self.x2, 0.0), w), min(max(self.y2, 0.0), h))


@dataclass(frozen=True)
class Detection:
    bbox: BBox
    conf: float
    cls: int
    label: str
    track_id: Optional[int] = None
    source: str = "yolo"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


@dataclass
class FrameResult:
    frame_id: int
    detections: List[Detection] = field(default_factory=list)
    confirmed: List[Detection] = field(default_factory=list)
    alerts: List["AlertEvent"] = field(default_factory=list)
    latency_ms: float = 0.0
    source: str = ""
    evidence_level: EvidenceLevel = EvidenceLevel.MEASURED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "detections": [d.to_dict() for d in self.detections],
            "confirmed": [d.to_dict() for d in self.confirmed],
            "alerts": [a.to_dict() for a in self.alerts],
            "latency_ms": round(self.latency_ms, 3),
            "source": self.source,
            "evidence_level": self.evidence_level.value,
        }


@dataclass
class AlertEvent:
    level: AlertLevel
    message: str
    frame_id: int
    track_ids: Tuple[int, ...] = ()
    count_without_helmet: int = 0
    snapshot_path: str = ""
    ts: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {**asdict(self), "level": self.level.value,
                "track_ids": list(self.track_ids)}


@dataclass(frozen=True)
class EvalMetrics:
    """Detection metrics on a labelled, held-out split."""
    precision: float
    recall: float
    map50: float
    map50_95: float
    per_class: Dict[str, Dict[str, float]]
    n_images: int
    n_gt: int
    inference_ms_per_image: float
    fps: float
    evidence_level: EvidenceLevel = EvidenceLevel.MEASURED
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["evidence_level"] = self.evidence_level.value
        return d


@dataclass(frozen=True)
class CocoAnnotation:
    image_id: int
    category_id: int
    bbox: BBox
    annotation_id: int
    is_crowd: int = 0
