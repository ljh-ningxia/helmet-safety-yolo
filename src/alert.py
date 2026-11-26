"""创新点 ③ —— 分级预警与可回溯事件闭环。

【真实问题】"检测到了"不等于"现场被处置了"。多数课程项目止步于画框，
          没有把结果接到「通讯/报表 → 业务动作」，也没有可复核的事件留存。
【改动环节】补齐基线流程的最后两环，并加入告警去重冷却。
【做法】L1 提示 / L2 预警 / L3 严重 三级；同一轨迹在冷却窗口内只产生一条事件；
        每条事件保存 JSONL 记录 + 证据快照图；可选 Webhook 推送到现场看板/工单系统。
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from .schema import Detection

LEVELS = {1: "L1 提示", 2: "L2 预警", 3: "L3 严重"}


@dataclass
class AlertEvent:
    ts: float
    level: int
    level_name: str
    track_id: str
    box: List[float]
    conf: float
    count_in_frame: int
    message: str
    snapshot: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.ts)),
            "ts": round(self.ts, 3),
            "level": self.level_name,
            "track_id": self.track_id,
            "box": [round(v, 1) for v in self.box],
            "conf": round(self.conf, 3),
            "count_in_frame": self.count_in_frame,
            "message": self.message,
            "snapshot": self.snapshot,
        }


class AlertManager:
    def __init__(self, log_dir: str = "experiments/results", cooldown: float = 8.0,
                 enable_dedup: bool = True, webhook: str = "",
                 save_snapshot: bool = True):
        self.log_dir = log_dir
        self.cooldown = cooldown
        self.enable_dedup = enable_dedup
        self.webhook = webhook
        self.save_snapshot = save_snapshot
        self._last: Dict[str, float] = {}
        self.events: List[AlertEvent] = []
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, "alerts.jsonl")

    def reset(self):
        self._last.clear()
        self.events.clear()

    @staticmethod
    def _level(n_violations: int) -> int:
        if n_violations >= 3:
            return 3
        if n_violations >= 2:
            return 2
        return 1

    def emit(self, confirmed: List[Detection], frame=None, frame_id: int = 0) -> List[AlertEvent]:
        """把「已确认违规」转成事件。返回本帧新增事件（去重后）。"""
        if not confirmed:
            return []
        level = self._level(len(confirmed))
        new_events: List[AlertEvent] = []
        now = time.time()
        for d in confirmed:
            key = d.source or "anon"
            if self.enable_dedup and (now - self._last.get(key, -1e9)) < self.cooldown:
                continue
            self._last[key] = now
            ev = AlertEvent(
                ts=now, level=level, level_name=LEVELS[level],
                track_id=key, box=d.as_list(), conf=d.conf,
                count_in_frame=len(confirmed),
                message=f"检测到 {len(confirmed)} 人未佩戴安全帽，请现场核查（帧 {frame_id}）",
            )
            if self.save_snapshot and frame is not None:
                ev.snapshot = self._snapshot(frame, d, frame_id)
            new_events.append(ev)
            self.events.append(ev)
            self._append(ev)
            if self.webhook:
                self._push(ev)
        return new_events

    def _append(self, ev: AlertEvent):
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(ev.to_dict(), ensure_ascii=False) + "\n")

    def _snapshot(self, frame, d: Detection, frame_id: int) -> str:
        try:
            import cv2
            import numpy as np
            snap_dir = os.path.join(self.log_dir, "snapshots")
            os.makedirs(snap_dir, exist_ok=True)
            img = frame.copy()
            cv2.rectangle(img, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)),
                          (0, 0, 230), 2)
            path = os.path.join(snap_dir, f"alert_{frame_id}_{int(time.time()*1000)}.jpg")
            cv2.imwrite(path, img)
            return path
        except Exception:
            return ""

    def _push(self, ev: AlertEvent):
        """可选 Webhook。默认关闭：对外动作必须本人确认后才启用（课程规范：外部动作需谨慎）。"""
        try:
            import urllib.request
            req = urllib.request.Request(
                self.webhook, data=json.dumps(ev.to_dict(), ensure_ascii=False).encode(),
                headers={"Content-Type": "application/json"}, method="POST")
            urllib.request.urlopen(req, timeout=2).close()
        except Exception:
            pass

    def summary(self) -> Dict[str, Any]:
        return {
            "total_events": len(self.events),
            "by_level": {LEVELS[k]: sum(1 for e in self.events if e.level == k)
                         for k in LEVELS},
            "dedup_enabled": self.enable_dedup,
            "cooldown_sec": self.cooldown,
            "log_path": self.log_path,
        }


def describe() -> dict:
    return {
        "module": "alert",
        "change": "补齐通讯/报表 → 业务动作闭环；新增分级 + 冷却去重 + 证据快照",
        "unchanged": ["检测器", "判定逻辑", "阈值", "数据集", "硬件"],
        "expected_cost": "增加磁盘/网络开销；冷却窗口内同一目标不再重复上报",
    }
