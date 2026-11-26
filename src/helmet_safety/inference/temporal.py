"""Lightweight IoU tracker + temporal majority voting.

A frame-level detector flickers: a head may be visible in 4 of 5 frames but
missed in one. We associate boxes across frames by IoU and only emit a
violation once a track accumulates ``min_votes`` hits within ``window`` frames.

This is deliberately a small, auditable tracker — no external dependency, no
hidden state — so its effect can be isolated in a single-variable A/B test.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List

from ..schemas import BBox, Detection

VIOLATION_CLS = 1  # no-hardhat


@dataclass
class Track:
    tid: int
    bbox: BBox
    cls: int
    hits: deque = field(default_factory=deque)
    last_frame: int = 0
    best_conf: float = 0.0


class TemporalVoter:
    def __init__(self, window: int = 5, min_votes: int = 3,
                 iou: float = 0.45, max_age: int = 3):
        if min_votes > window:
            raise ValueError("min_votes cannot exceed window")
        self.window, self.min_votes, self.iou, self.max_age = window, min_votes, iou, max_age
        self.tracks: Dict[int, Track] = {}
        self._next = 1
        self.frame_id = 0

    def reset(self) -> None:
        self.tracks.clear()
        self._next = 1
        self.frame_id = 0

    def update(self, detections: List[Detection]) -> List[Detection]:
        self.frame_id += 1
        fid = self.frame_id
        for t in self.tracks.values():
            t.hits.append(0)
            if len(t.hits) > self.window:
                t.hits.popleft()

        unmatched = list(detections)
        for t in list(self.tracks.values()):
            if fid - t.last_frame > self.max_age:
                self.tracks.pop(t.tid, None)
                continue
            best_i, best_iou = -1, 0.0
            for i, d in enumerate(unmatched):
                if d.cls != t.cls:
                    continue
                v = t.bbox.iou(d.bbox)
                if v > best_iou:
                    best_i, best_iou = i, v
            if best_i >= 0 and best_iou >= self.iou:
                d = unmatched.pop(best_i)
                t.bbox, t.last_frame = d.bbox, fid
                t.best_conf = max(t.best_conf, d.conf)
                if t.hits:
                    t.hits[-1] = 1
                else:
                    t.hits.append(1)

        for d in unmatched:
            t = Track(self._next, d.bbox, d.cls, deque([1], maxlen=self.window), fid, d.conf)
            self.tracks[self._next] = t
            self._next += 1

        out: List[Detection] = []
        for t in self.tracks.values():
            if sum(t.hits) >= self.min_votes:
                out.append(Detection(t.bbox, t.best_conf, t.cls,
                                     "no-hardhat" if t.cls == VIOLATION_CLS else "hardhat",
                                     track_id=t.tid, source="temporal"))
        return out
