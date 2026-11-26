"""统一数据结构。全工程只使用这几种对象交换数据，方便替换检测后端。"""

from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class Detection:
    """一个检测框。坐标统一为原图像素坐标系 (x1, y1, x2, y2)。"""
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    cls: int                 # 0 = helmet, 1 = no_helmet
    label: str = "helmet"
    source: str = "full"     # full | tile:r,c | temporal | demo

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    def as_list(self) -> List[float]:
        return [self.x1, self.y1, self.x2, self.y2]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "box": [round(v, 1) for v in self.as_list()],
            "conf": round(self.conf, 3),
            "cls": self.cls,
            "label": self.label,
            "source": self.source,
        }


@dataclass
class FrameResult:
    """单帧处理结果：完整保留判定链路，便于事后复核。"""
    frame_id: int
    detections: List[Detection] = field(default_factory=list)
    confirmed: List[Detection] = field(default_factory=list)   # 时序投票确认后的违规
    alerts: List[Dict[str, Any]] = field(default_factory=list)
    latency_ms: float = 0.0
    backend: str = ""
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "detections": [d.to_dict() for d in self.detections],
            "confirmed": [d.to_dict() for d in self.confirmed],
            "alerts": self.alerts,
            "latency_ms": round(self.latency_ms, 2),
            "backend": self.backend,
        }
