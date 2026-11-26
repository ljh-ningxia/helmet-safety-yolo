"""统一数据源。所有来源都产出 (frame, gts_or_None)，保证 A/B 用同一输入。"""

from __future__ import annotations

import os
from typing import Iterator, List, Optional, Tuple

import cv2
import numpy as np

from .schema import Detection
from .synthetic import SyntheticSite


def _read_yolo_label(path: str, w: int, h: int) -> List[Detection]:
    out = []
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) != 5:
                continue
            cls, cx, cy, bw, bh = int(p[0]), *[float(v) for v in p[1:]]
            x1, y1 = (cx - bw / 2) * w, (cy - bh / 2) * h
            x2, y2 = (cx + bw / 2) * w, (cy + bh / 2) * h
            out.append(Detection(x1, y1, x2, y2, 1.0, cls,
                                 "helmet" if cls == 0 else "no_helmet", "gt"))
    return out


class FrameSource:
    """source 可为：synthetic / 图片路径 / 视频路径 / 摄像头编号(0,1...) / 数据集目录。"""

    def __init__(self, source: str = "synthetic", frames: int = 90,
                 seed: int = 20261126, labels_dir: Optional[str] = None):
        self.kind = "synthetic" if source == "synthetic" else self._classify(source)
        self.source = source
        self.frames = frames
        self.seed = seed
        self.labels_dir = labels_dir
        self._cap = None

    @staticmethod
    def _classify(s: str) -> str:
        if s.isdigit():
            return "camera"
        if os.path.isdir(s):
            return "dataset"
        ext = os.path.splitext(s)[1].lower()
        if ext in (".jpg", ".jpeg", ".png", ".bmp"):
            return "image"
        return "video"

    def __iter__(self) -> Iterator[Tuple[np.ndarray, Optional[List[Detection]]]]:
        if self.kind == "synthetic":
            site = SyntheticSite(seed=self.seed)
            for i in range(self.frames):
                yield site.frame(i)
            return
        if self.kind == "image":
            img = cv2.imread(self.source)
            if img is None:
                raise FileNotFoundError(self.source)
            lab = os.path.splitext(self.source)[0] + ".txt"
            yield img, _read_yolo_label(lab, img.shape[1], img.shape[0])
            return
        if self.kind == "dataset":
            files = sorted(f for f in os.listdir(self.source)
                           if f.lower().endswith((".jpg", ".jpeg", ".png")))
            for name in files[:self.frames]:
                img = cv2.imread(os.path.join(self.source, name))
                if img is None:
                    continue
                lab = os.path.join(self.labels_dir or self.source,
                                   os.path.splitext(name)[0] + ".txt")
                yield img, _read_yolo_label(lab, img.shape[1], img.shape[0])
            return
        # video / camera
        target = int(self.source) if self.kind == "camera" else self.source
        cap = cv2.VideoCapture(target)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频源：{self.source}")
        self._cap = cap
        n = 0
        while True:
            ok, frame = cap.read()
            if not ok or (self.kind != "camera" and n >= self.frames):
                break
            n += 1
            yield frame, None      # 视频/摄像头无真值 → 只做描述性统计
        cap.release()

    def describe(self) -> dict:
        return {"kind": self.kind, "source": self.source,
                "max_frames": self.frames, "seed": self.seed,
                "has_ground_truth": self.kind in ("synthetic", "image", "dataset")}
