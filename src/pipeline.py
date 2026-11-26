"""系统基线：成像 → 预处理 → 定位/检测 → 判定 → 通讯/报表 → 业务动作。

三个开关各自对应「唯一主要变动」，A/B 实验每次只打开其中一个：
    enable_tiling        → 改动定位/检测环节的推理组织方式
    enable_temporal      → 改动判定环节
    enable_alert_dedup   → 改动业务动作环节
"""

from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from . import tiling, temporal
from .alert import AlertManager, describe as alert_describe
from .detector import BaseDetector
from .schema import Detection, FrameResult


@dataclass
class PipelineConfig:
    # —— 基线（A）固定不动的条件 ——
    conf: float = 0.25
    iou: float = 0.45
    imgsz: int = 640
    # —— 唯一主要变动（B），默认全关 = 基线 A ——
    enable_tiling: bool = False
    enable_temporal: bool = False
    enable_alert_dedup: bool = False
    # —— 各模块内部参数（一旦冻结，A/B 两边一致）——
    tile_size: int = 640
    tile_overlap: float = 0.25
    tile_scale: float = 2.0
    hit_k: int = 3
    miss_k: int = 3
    cooldown_sec: float = 8.0
    # —— 运行参数 ——
    save_alerts: bool = True
    log_dir: str = "experiments/results"

    def variant(self, name: str) -> "PipelineConfig":
        """从基线派生一个只改一项的变体。这是公平 A/B 的约束入口。"""
        c = PipelineConfig(**asdict(self))
        if name == "A":
            c.enable_tiling = c.enable_temporal = c.enable_alert_dedup = False
        elif name == "B_tiling":
            c.enable_tiling = True
        elif name == "B_temporal":
            c.enable_temporal = True
        elif name == "B_dedup":
            c.enable_alert_dedup = True
        else:
            raise ValueError(f"未知变体 {name}")
        return c

    def changed_modules(self) -> List[str]:
        out = []
        if self.enable_tiling:
            out.append("定位/检测：整图推理 → 重叠切片推理")
        if self.enable_temporal:
            out.append("判定：单帧即判 → 连续 K 帧投票")
        if self.enable_alert_dedup:
            out.append("业务动作：逐帧上报 → 冷却去重 + 分级 + 证据留存")
        return out or ["无（基线 A）"]


class SafetyPipeline:
    def __init__(self, detector: BaseDetector, config: PipelineConfig,
                 collect: bool = True):
        self.detector = detector
        self.cfg = config
        self.voter = temporal.TemporalVoter(hit_k=config.hit_k,
                                            miss_k=config.miss_k,
                                            match_iou=config.iou)
        self.alerts = AlertManager(log_dir=config.log_dir,
                                   cooldown=config.cooldown_sec,
                                   enable_dedup=config.enable_alert_dedup,
                                   save_snapshot=config.save_alerts)
        self.collect = collect
        self.results: List[FrameResult] = []
        self.frame_id = 0

    def reset(self):
        self.voter.reset()
        self.alerts.reset()
        self.results.clear()
        self.frame_id = 0
        if hasattr(self.detector, "reset"):
            self.detector.reset()

    def process(self, frame: np.ndarray, emit: bool = True) -> FrameResult:
        t0 = time.perf_counter()
        self.frame_id += 1
        cfg = self.cfg
        original = frame

        # ① 成像（由 datasource 提供）—— ② 预处理：仅等比限幅，A/B 完全一致
        h, w = frame.shape[:2]
        if max(h, w) > 1920:
            k = 1920 / max(h, w)
            frame = cv2.resize(frame, None, fx=k, fy=k)
            scale_back = 1.0 / k
        else:
            scale_back = 1.0

        # ③ 定位/检测
        if cfg.enable_tiling:
            dets = tiling.tiled_predict(self.detector, frame, cfg.tile_size,
                                        cfg.tile_overlap, cfg.tile_scale, cfg.iou)
        else:
            dets = self.detector.predict(frame)
        if scale_back != 1.0:
            for d in dets:
                d.x1, d.y1, d.x2, d.y2 = [v * scale_back for v in d.as_list()]

        # ④ 判定
        if cfg.enable_temporal:
            confirmed = self.voter.update(dets)
        else:
            confirmed = [d for d in dets if d.cls == 1]

        # ⑤ 通讯/报表 + ⑥ 业务动作
        events = self.alerts.emit(confirmed, original, self.frame_id) if emit else []
        r = FrameResult(
            frame_id=self.frame_id,
            detections=dets,
            confirmed=confirmed,
            alerts=[e.to_dict() for e in events],
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            backend=self.detector.name,
            config=asdict(cfg),
        )
        if self.collect:
            self.results.append(r)
        return r

    def describe(self) -> Dict[str, Any]:
        return {
            "backend": self.detector.describe(),
            "config": asdict(self.cfg),
            "changed_modules": self.cfg.changed_modules(),
            "innovations": [tiling.describe(), temporal.describe(), alert_describe()],
        }
