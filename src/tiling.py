"""创新点 ① —— 方向自适应的重叠切片推理（Tiled Inference）。

【真实问题】工地监控多为高位俯拍远景，安全帽在 1280×720 画面里常常只有
          20×20 像素；整图缩放到 640 输入后目标被进一步压缩，容易漏检。
【改动环节】只改「定位/检测」这一步的推理组织方式，不改模型权重、不改阈值。
【做法】把原图切成带重叠的切片，逐片放大后推理，把框映射回原图坐标，
        最后用跨切片 NMS 融合重叠框。
【代价】推理次数增加 → 时延/算力上升；这是必须在实验里同时测的成本项。
"""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

from .schema import Detection


def _nms(dets: List[Detection], iou_thr: float) -> List[Detection]:
    """按类别分别做 NMS，置信度优先。"""
    kept: List[Detection] = []
    for cls in (0, 1):
        group = sorted([d for d in dets if d.cls == cls], key=lambda d: -d.conf)
        while group:
            best = group.pop(0)
            kept.append(best)
            group = [d for d in group if _iou(best, d) < iou_thr]
    return sorted(kept, key=lambda d: -d.conf)


def _iou(a: Detection, b: Detection) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def _slices(w: int, h: int, size: int, overlap: float) -> List[Tuple[int, int, int, int]]:
    stride = max(64, int(size * (1 - overlap)))
    xs, ys = [], []
    x = 0
    while x < w:
        xs.append(min(x, max(0, w - size)))
        if x + size >= w:
            break
        x += stride
    y = 0
    while y < h:
        ys.append(min(y, max(0, h - size)))
        if y + size >= h:
            break
        y += stride
    # 去重（当图很小时可能重复同一坐标）
    return list(dict.fromkeys((xx, yy, min(xx + size, w), min(yy + size, h))
                              for yy in ys for xx in xs))


def tiled_predict(detector, frame: np.ndarray, size: int = 640,
                  overlap: float = 0.25, scale: float = 1.5,
                  nms_iou: float = 0.45) -> List[Detection]:
    """切片推理主入口。返回原图坐标系下的检测框。

    scale > 1 表示切片放大后再推理，等价于给小目标更高的有效输入分辨率。
    """
    h, w = frame.shape[:2]
    boxes = _slices(w, h, size, overlap)
    result: List[Detection] = []
    for r, (x1, y1, x2, y2) in enumerate(boxes):
        tile = frame[y1:y2, x1:x2]
        if scale != 1.0:
            tile = cv2.resize(tile, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_LINEAR)
            k = 1.0 / scale
        else:
            k = 1.0
        for d in detector.predict(tile):
            result.append(Detection(
                x1 + d.x1 * k, y1 + d.y1 * k,
                x1 + d.x2 * k, y1 + d.y2 * k,
                d.conf, d.cls, d.label, source=f"tile{r}",
            ))
    # 整图一路也保留，避免大目标在切片边界被切断
    for d in detector.predict(frame):
        result.append(Detection(d.x1, d.y1, d.x2, d.y2, d.conf, d.cls,
                                d.label, source="full"))
    return _nms(result, nms_iou)


def describe() -> dict:
    return {
        "module": "tiling",
        "change": "仅推理组织方式：整图 → 重叠切片 + 放大 + 跨切片 NMS",
        "unchanged": ["模型权重", "置信度阈值", "NMS IoU", "数据集划分", "硬件"],
        "expected_cost": "推理次数与显存/CPU 占用上升，FPS 下降",
    }
