"""检测后端。

后端 A（正式）：YOLODetector —— 基于 ultralytics YOLOv8，加载用户自行训练的
                 best.pt（类别 0=helmet, 1=no_helmet）。项目不内置任何权重，
                 因为权重必须由本人训练或来自可追溯来源，不能凭空声明。
后端 B（演练）：DemoDetector —— 仅用于「能看、能跑、能对比、能复述」的链路
                 演练。它基于 HSV 颜色分割产生确定性输出，并带有可配置的
                 小目标面积门槛，用来复现「远景小目标被尺度压缩」这一现象。
                 ⚠️ DemoDetector 的一切数值均为 SELFTEST_ONLY，禁止写入简历、
                 答辩结论或作为模型性能证据。
"""

from __future__ import annotations

import abc
import hashlib
import random
from typing import List, Dict, Any

import cv2
import numpy as np

from .schema import Detection

EVIDENCE_LEVEL = {
    "yolo": "MEASURED",        # 需要真实权重 + 带真值测试集
    "demo": "SELFTEST_ONLY",   # 仅流程自检，不是性能证据
}


class BaseDetector(abc.ABC):
    name = "base"

    @abc.abstractmethod
    def predict(self, frame: np.ndarray) -> List[Detection]:
        ...

    def describe(self) -> Dict[str, Any]:
        return {"backend": self.name, "evidence_level": EVIDENCE_LEVEL.get(self.name, "UNKNOWN")}


class YOLODetector(BaseDetector):
    """正式后端。未安装 ultralytics 或未给出权重时会明确报错，绝不静默降级。"""

    name = "yolo"

    def __init__(self, weights: str, conf: float = 0.25, iou: float = 0.45,
                 device: str = "auto", imgsz: int = 640):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "正式后端需要 ultralytics：pip install ultralytics；"
                "若只想演练链路，请使用 --backend demo（结果不可作为性能证据）。"
            ) from exc
        self.model = YOLO(weights)
        self.conf, self.iou, self.device, self.imgsz = conf, iou, device, imgsz
        self.weights = weights

    def predict(self, frame: np.ndarray) -> List[Detection]:
        results = self.model.predict(
            source=frame, conf=self.conf, iou=self.iou, imgsz=self.imgsz,
            device=None if self.device == "auto" else self.device, verbose=False,
        )
        out: List[Detection] = []
        for r in results:
            boxes = getattr(r, "boxes", None)
            if boxes is None:
                continue
            for b in boxes:
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
                cls = int(b.cls[0].item())
                out.append(Detection(x1, y1, x2, y2, float(b.conf[0].item()),
                                     cls, "helmet" if cls == 0 else "no_helmet"))
        return out

    def describe(self) -> Dict[str, Any]:
        d = super().describe()
        d.update(weights=self.weights, conf=self.conf, iou=self.iou, imgsz=self.imgsz)
        return d


class DemoDetector(BaseDetector):
    """确定性演练后端（HSV 颜色分割）。

    合成场景中：黄色/红色/蓝色安全帽 = helmet；肤色头部 = no_helmet。
    min_area 模拟检测器对远景小目标的失效：小于该面积的框会被丢弃。
    切片推理把切片放大后再送入这里，因此能真实地改变召回（用于演示 A/B 链路）。
    """

    name = "demo"

    # HSV 范围 (H, S, V)，OpenCV H ∈ [0, 180]
    HELMET_MASKS = [
        ((18, 90, 90), (38, 255, 255)),    # 黄
        ((0, 100, 90), (10, 255, 255)),    # 红
        ((100, 80, 80), (130, 255, 255)),  # 蓝
    ]
    SKIN_MASK = ((0, 30, 90), (25, 170, 255))

    def __init__(self, conf: float = 0.25, min_area: int = 300,
                 min_wh: int = 14, seed: int = 20261126):
        self.conf = conf
        self.min_area = min_area
        self.min_wh = min_wh
        self.seed = seed
        self._rng = random.Random(seed)

    def reset(self):
        self._rng = random.Random(self.seed)

    def _blobs(self, frame: np.ndarray, ranges) -> List[Detection]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in ranges:
            mask |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        out: List[Detection] = []
        for i in range(1, n):
            x, y, w, h, area = [int(v) for v in stats[i]]
            if area < self.min_area or w < self.min_wh or h < self.min_wh:
                continue
            conf = min(0.99, 0.55 + 0.4 * min(1.0, area / 6000.0))
            out.append(Detection(float(x), float(y), float(x + w), float(y + h),
                                 round(conf, 3), 0, "helmet"))
        return out

    def predict(self, frame: np.ndarray) -> List[Detection]:
        helmets = self._blobs(frame, self.HELMET_MASKS)
        heads = self._blobs(frame, [self.SKIN_MASK])
        # 被安全帽覆盖的头部不算违规：用 IoU 抑制
        violations = []
        for h in heads:
            covered = any(_iou(h, k) > 0.15 for k in helmets)
            if covered:
                continue
            d = Detection(h.x1, h.y1, h.x2, h.y2, h.conf, 1, "no_helmet")
            violations.append(d)
        return helmets + violations

    def describe(self) -> Dict[str, Any]:
        d = super().describe()
        d.update(min_area=self.min_area, min_wh=self.min_wh, seed=self.seed,
                 note="HSV 颜色分割演练器，非 YOLO 模型；数值不可作为性能证据")
        return d


def _iou(a: Detection, b: Detection) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def build_detector(backend: str, **kwargs) -> BaseDetector:
    if backend == "yolo":
        return YOLODetector(**kwargs)
    if backend == "demo":
        return DemoDetector(**{k: v for k, v in kwargs.items()
                               if k in ("conf", "min_area", "min_wh", "seed")})
    raise ValueError(f"未知后端 {backend}，可选 yolo / demo")


def environment_fingerprint() -> Dict[str, Any]:
    """冻结 A/B 实验条件：让「同一硬件、同一版本」可复核。"""
    import platform
    import sys
    raw = "|".join([platform.platform(), platform.processor(),
                    sys.version, cv2.__version__, np.__version__])
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "threads": cv2.getNumThreads(),
        "hash": hashlib.sha256(raw.encode()).hexdigest()[:16],
    }
