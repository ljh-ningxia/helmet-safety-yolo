"""评估指标。严格区分「有真值」与「无真值」两种情形。

课程规范反复强调：检出框数量变化 ≠ 准确率提升。
因此本模块在没有人工真值时拒绝输出 Precision / Recall / mAP，只输出
「检测输出统计」，并在报告中显式标注 insufficient_ground_truth=True。
"""

from __future__ import annotations

from typing import List, Dict, Any, Sequence, Tuple

from .schema import Detection


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ub = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = ua + ub - inter
    return inter / union if union > 0 else 0.0


def match_by_class(preds: List[Detection], gts: List[Detection],
                   iou_thr: float = 0.5) -> Dict[int, Dict[str, Any]]:
    """按类别做贪心匹配，返回每类 TP/FP/FN 与置信度序列（供 AP 计算）。"""
    result: Dict[int, Dict[str, Any]] = {}
    for cls in (0, 1):
        p = sorted([d for d in preds if d.cls == cls], key=lambda x: -x.conf)
        g = [d for d in gts if d.cls == cls]
        used = set()
        tp_flags, confs = [], []
        for d in p:
            best, best_iou = -1, iou_thr
            for j, gt in enumerate(g):
                if j in used:
                    continue
                v = iou(d.as_list(), gt.as_list())
                if v > best_iou:
                    best, best_iou = j, v
            if best >= 0:
                used.add(best)
                tp_flags.append(1)
            else:
                tp_flags.append(0)
            confs.append(d.conf)
        tp, fp = sum(tp_flags), len(tp_flags) - sum(tp_flags)
        fn = len(g) - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        result[cls] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "ap50": round(_average_precision(tp_flags), 4),
        }
    return result


def _average_precision(tp_flags: List[int]) -> float:
    if not tp_flags:
        return 0.0
    tp = 0
    curve = []
    for i, f in enumerate(tp_flags, 1):
        tp += f
        curve.append(tp / i)
    return sum(curve) / len(curve)


def evaluate(preds_per_frame: List[List[Detection]],
             gts_per_frame: List[List[Detection]] | None,
             latency_ms: List[float],
             failures: int = 0) -> Dict[str, Any]:
    """汇总一次实验。gts_per_frame 为 None 时只给描述性统计。"""
    n = len(preds_per_frame)
    total_ms = sum(latency_ms)
    base: Dict[str, Any] = {
        "frames": n,
        "total_detections": sum(len(p) for p in preds_per_frame),
        "mean_detections_per_frame": round(sum(len(p) for p in preds_per_frame) / n, 3) if n else 0,
        "violation_detections": sum(sum(1 for d in p if d.cls == 1) for p in preds_per_frame),
        "mean_latency_ms": round(total_ms / n, 2) if n else 0,
        "fps": round(1000.0 * n / total_ms, 2) if total_ms > 0 else 0,
        "failure_count": failures,
        "failure_rate": round(failures / n, 4) if n else 0,
    }
    if gts_per_frame is None:
        base.update({
            "ground_truth": False,
            "insufficient_ground_truth": True,
            "precision": None, "recall": None, "map50": None,
            "claim_limit": "无人工真值，仅能描述检测输出数量与开销变化，不可推断准确率/召回率改善",
        })
        return base

    flat_pred = [d for frame in preds_per_frame for d in frame]
    flat_gt = [d for frame in gts_per_frame for d in frame]
    per_cls = match_by_class(flat_pred, flat_gt)
    base.update({
        "ground_truth": True,
        "insufficient_ground_truth": False,
        "per_class": per_cls,
        "precision": round((per_cls[0]["precision"] + per_cls[1]["precision"]) / 2, 4),
        "recall": round((per_cls[0]["recall"] + per_cls[1]["recall"]) / 2, 4),
        "map50": round((per_cls[0]["ap50"] + per_cls[1]["ap50"]) / 2, 4),
        "claim_limit": "指标仅在给定真值测试集与冻结条件下成立；跨数据集/跨硬件不可外推",
    })
    return base


def compare(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """只做可解释的差值，不生成「提升 xx%」这类未经核实的营销话术。"""
    keys = ["total_detections", "violation_detections", "mean_latency_ms",
            "fps", "failure_rate", "precision", "recall", "map50"]
    out = {}
    for k in keys:
        va, vb = a.get(k), b.get(k)
        out[k] = {"A": va, "B": vb,
                  "delta": round(vb - va, 4) if isinstance(va, (int, float))
                  and isinstance(vb, (int, float)) else None}
    out["note"] = "检出框数量变化不等于准确率提升；只有 ground_truth=True 时 precision/recall/map50 可解释"
    return out
