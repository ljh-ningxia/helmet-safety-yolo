"""创新点 ② —— 跨帧时序投票（Temporal Voting）。

【真实问题】单帧误检/遮挡会让告警忽明忽暗：一次闪光就报一次警，
          现场人员很快对告警脱敏，同时报表里全是重复事件。
【改动环节】只改「判定」这一步：从「单帧即判」改为「连续 K 帧确认」。
【做法】用IoU 做轻量跨帧关联（不引入额外跟踪模型），为每个目标维护
        命中/未命中计数；连续 hit_k 帧确认才输出违规，miss_k 帧未命中释放。
【代价】引入 K 帧固有判定延迟；快速通过画面的目标可能来不及确认。
"""

from __future__ import annotations

from typing import List, Dict

from .schema import Detection


def iou(a: Detection, b: Detection) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


class Track:
    __slots__ = ("box", "cls", "hits", "misses", "confirmed", "tid", "last_conf")

    def __init__(self, tid: int, d: Detection):
        self.box = d
        self.cls = d.cls
        self.hits = 1
        self.misses = 0
        self.confirmed = False
        self.tid = tid
        self.last_conf = d.conf


class TemporalVoter:
    """跨帧投票器。只处理 cls=1（未佩戴安全帽）的违规候选。"""

    def __init__(self, hit_k: int = 3, miss_k: int = 3, match_iou: float = 0.30):
        self.hit_k = hit_k
        self.miss_k = miss_k
        self.match_iou = match_iou
        self.tracks: Dict[int, Track] = {}
        self._next_id = 1
        self.latency_frames = hit_k - 1

    def reset(self):
        self.tracks.clear()
        self._next_id = 1

    def update(self, detections: List[Detection]) -> List[Detection]:
        candidates = [d for d in detections if d.cls == 1]
        used = set()
        # 1) 关联已有轨迹
        for t in self.tracks.values():
            best, best_iou = None, self.match_iou
            for i, d in enumerate(candidates):
                if i in used:
                    continue
                v = iou(t.box, d)
                if v > best_iou:
                    best, best_iou = i, v
            if best is None:
                t.misses += 1
                t.hits = 0
            else:
                d = candidates[best]
                used.add(best)
                t.box, t.last_conf = d, d.conf
                t.misses = 0
                t.hits += 1
                if t.hits >= self.hit_k:
                    t.confirmed = True
        # 2) 新建轨迹
        for i, d in enumerate(candidates):
            if i in used:
                continue
            t = Track(self._next_id, d)
            self._next_id += 1
            t.confirmed = self.hit_k <= 1
            self.tracks[t.tid] = t
        # 3) 清理
        for tid in [k for k, v in self.tracks.items() if v.misses >= self.miss_k]:
            del self.tracks[tid]
        # 4) 输出已确认轨迹（每帧输出一次，由 AlertManager 负责去重）
        out = []
        for t in self.tracks.values():
            if t.confirmed:
                d = t.box
                out.append(Detection(d.x1, d.y1, d.x2, d.y2, t.last_conf,
                                     d.cls, d.label, source=f"track{t.tid}"))
        return out


def describe() -> dict:
    return {
        "module": "temporal",
        "change": "仅判定策略：单帧即报 → 连续 K 帧命中确认",
        "unchanged": ["检测器与权重", "阈值", "数据集", "硬件", "输入帧序列"],
        "expected_cost": "引入 K-1 帧判定延迟；快速通过目标可能漏报",
    }
