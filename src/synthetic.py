"""可复现的合成工地场景生成器（仅用于流程自检 / 界面演练）。

⚠️ 重要声明：
  合成图不等于真实工地数据。用它跑出的任何指标都只是「链路是否跑通」的
  自检结果，evidence_level = SELFTEST_ONLY，禁止写入简历、答辩结论、
  或对外声称的模型精度。真实性能必须用带人工真值的实拍测试集重跑。

生成内容：1280×720 俯视工地画面，含近景/远景人物；
        佩戴安全帽 → helmet(0)，未佩戴 → no_helmet(1)；
        同时输出逐帧真值框，供 A/B 实验计算 P/R/mAP。
"""

from __future__ import annotations

import json
import os
import random
from typing import List, Tuple

import cv2
import numpy as np

from .schema import Detection

W, H = 1280, 720
HELMET_BGR = [(30, 220, 240), (40, 40, 220), (200, 90, 30)]   # 黄 / 红 / 蓝
SKIN_BGR = (120, 165, 205)
BODY_BGR = [(60, 80, 60), (50, 60, 110), (90, 90, 95)]


class SyntheticSite:
    """确定性帧序列。相同 seed + frame_id 永远得到同一帧。"""

    def __init__(self, seed: int = 20261126, n_workers: int = 7,
                 far_ratio: float = 0.6, violation_ratio: float = 0.35):
        self.seed = seed
        self.n_workers = n_workers
        self.far_ratio = far_ratio
        self.violation_ratio = violation_ratio

    def _layout(self, frame_id: int):
        rng = random.Random(self.seed + frame_id // 30)  # 每 30 帧换一次布局
        workers = []
        for i in range(self.n_workers):
            far = rng.random() < self.far_ratio
            size = rng.randint(16, 26) if far else rng.randint(52, 86)
            x = rng.randint(90, W - 110)
            y = rng.randint(120, H - 90) if not far else rng.randint(90, 260)
            has_helmet = rng.random() > self.violation_ratio
            color = HELMET_BGR[rng.randrange(len(HELMET_BGR))]
            workers.append((x, y, size, has_helmet, color))
        return workers

    def frame(self, frame_id: int) -> Tuple[np.ndarray, List[Detection]]:
        img = np.full((H, W, 3), (118, 122, 126), np.uint8)
        # 地面分区与施工线，给画面一点真实感
        cv2.rectangle(img, (60, 70), (W - 60, H - 50), (150, 155, 158), 2)
        for gx in range(120, W, 160):
            cv2.line(img, (gx, 70), (gx - 40, H - 50), (135, 140, 143), 1)
        cv2.line(img, (60, 300), (W - 60, 260), (170, 175, 90), 3)
        cv2.putText(img, "ZONE A - HELMET REQUIRED", (80, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 205, 210), 1, cv2.LINE_AA)

        rng = random.Random(self.seed + frame_id)
        gts: List[Detection] = []
        for (x, y, s, has_helmet, color) in self._layout(frame_id):
            # 每帧微小位移，制造时序连续性
            x += rng.randint(-2, 2)
            y += rng.randint(-2, 2)
            hh = int(s * 0.45)          # 头部/帽子高度
            bw = int(s * 0.9)
            # 身体
            cv2.rectangle(img, (x - bw // 2, y + hh), (x + bw // 2, y + s * 2),
                          BODY_BGR[(x // 100) % 3], -1)
            # 头部（肤色）
            cv2.rectangle(img, (x - s // 2, y), (x + s // 2, y + hh), SKIN_BGR, -1)
            if has_helmet:
                # 安全帽覆盖头顶，稍宽于头部
                cv2.rectangle(img, (x - s // 2 - 3, y - 3),
                              (x + s // 2 + 3, y + hh // 2 + 2), color, -1)
                gts.append(Detection(x - s / 2 - 3, y - 3, x + s / 2 + 3,
                                     y + hh / 2 + 2, 1.0, 0, "helmet", "gt"))
            else:
                gts.append(Detection(x - s / 2, y, x + s / 2, y + hh,
                                     1.0, 1, "no_helmet", "gt"))
        return img, gts

    def frames(self, count: int):
        for i in range(count):
            yield self.frame(i)


def export_dataset(out_dir: str, count: int = 60, seed: int = 20261126):
    """把合成帧导出成图片 + YOLO 标签格式，用于跑通训练/评估全流程。"""
    img_dir = os.path.join(out_dir, "images")
    lbl_dir = os.path.join(out_dir, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    site = SyntheticSite(seed=seed)
    manifest = []
    for i in range(count):
        img, gts = site.frame(i)
        p = os.path.join(img_dir, f"syn_{i:04d}.jpg")
        cv2.imwrite(p, img)
        lines = []
        for d in gts:
            cx = ((d.x1 + d.x2) / 2) / W
            cy = ((d.y1 + d.y2) / 2) / H
            bw = (d.x2 - d.x1) / W
            bh = (d.y2 - d.y1) / H
            lines.append(f"{d.cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        with open(os.path.join(lbl_dir, f"syn_{i:04d}.txt"), "w") as f:
            f.write("\n".join(lines))
        manifest.append({"image": p, "gt": [d.to_dict() for d in gts]})
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"seed": seed, "count": count,
                   "warning": "SELFTEST_ONLY - 合成数据，不是真实工地样本",
                   "frames": manifest}, f, ensure_ascii=False, indent=2)
    return out_dir
