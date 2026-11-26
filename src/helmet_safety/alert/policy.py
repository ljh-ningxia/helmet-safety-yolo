"""Alert policy, deduplication and pluggable sinks.

Business rules are configuration, not code buried in an if/else. The policy
takes a list of confirmed violations and returns zero or one AlertEvent.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Protocol

from ..logging import get_logger
from ..schemas import AlertEvent, AlertLevel, Detection

log = get_logger(__name__)


@dataclass(frozen=True)
class PolicyConfig:
    level_2_count: int = 2
    level_3_count: int = 4
    cooldown_frames: int = 30
    enable_dedup: bool = True
    sink_dir: str = "experiments/results"


class AlertSink(Protocol):
    def emit(self, event: AlertEvent) -> None: ...


class JsonlSink:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: AlertEvent) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")


class LogSink:
    def emit(self, event: AlertEvent) -> None:
        # NOTE: do not use `level=` as a structured field — it collides with
        # LoggerAdapter.log(level, msg, ...). Use `alert_level` instead.
        log.warning("alert", alert_level=event.level.value, frame=event.frame_id,
                    count=event.count_without_helmet, message=event.message)


class AlertManager:
    def __init__(self, cfg: PolicyConfig, sinks: Iterable[AlertSink] | None = None):
        self.cfg = cfg
        self.sinks: List[AlertSink] = list(sinks or [LogSink()])
        self._last_emitted: dict = {}
        self.frame_id = 0

    def reset(self) -> None:
        self._last_emitted.clear()
        self.frame_id = 0

    @staticmethod
    def _level(count: int, cfg: PolicyConfig) -> AlertLevel:
        if count >= cfg.level_3_count:
            return AlertLevel.L3
        if count >= cfg.level_2_count:
            return AlertLevel.L2
        return AlertLevel.L1

    def emit(self, confirmed: List[Detection]) -> List[AlertEvent]:
        self.frame_id += 1
        violations = [d for d in confirmed if d.cls == 1]
        if not violations:
            return []
        level = self._level(len(violations), self.cfg)
        key = (level, tuple(sorted(d.track_id or 0 for d in violations)))
        last = self._last_emitted.get(key, -10 ** 9)
        if self.cfg.enable_dedup and self.frame_id - last < self.cfg.cooldown_frames:
            return []
        self._last_emitted[key] = self.frame_id
        event = AlertEvent(
            level=level,
            message=f"检测到 {len(violations)} 人未佩戴安全帽，请现场核查",
            frame_id=self.frame_id,
            track_ids=tuple(d.track_id or 0 for d in violations),
            count_without_helmet=len(violations),
        )
        for sink in self.sinks:
            sink.emit(event)
        return [event]
