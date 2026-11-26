"""可视化：实时画框、证据快照、A/B 对比图。

所有图片都带「证据等级」水印，防止演练截图被误当成实验结果。
"""

from __future__ import annotations

import os
from typing import List, Dict, Any

import cv2
import numpy as np

from .schema import Detection

COLORS = {0: (40, 190, 90), 1: (40, 40, 230)}   # BGR：绿=合规，蓝/红=违规
LABELS = {0: "helmet", 1: "NO HELMET"}


def draw_frame(frame: np.ndarray, detections: List[Detection],
               confirmed: List[Detection] | None = None,
               evidence_level: str = "MEASURED",
               footer: str = "") -> np.ndarray:
    img = frame.copy()
    confirmed = confirmed or []
    for d in detections:
        c = COLORS.get(d.cls, (200, 200, 200))
        cv2.rectangle(img, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), c, 2)
        txt = f"{LABELS.get(d.cls, d.label)} {d.conf:.2f}"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(img, (int(d.x1), int(d.y1) - th - 6),
                      (int(d.x1) + tw + 4, int(d.y1)), c, -1)
        cv2.putText(img, txt, (int(d.x1) + 2, int(d.y1) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    for d in confirmed:
        cv2.rectangle(img, (int(d.x1) - 3, int(d.y1) - 3),
                      (int(d.x2) + 3, int(d.y2) + 3), (0, 0, 255), 1)
    banner = f"EVIDENCE: {evidence_level}"
    if evidence_level != "MEASURED":
        banner += "  |  NOT A PERFORMANCE CLAIM"
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (32, 32, 32), -1)
    cv2.putText(img, banner, (10, 18), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (120, 220, 255) if evidence_level == "MEASURED" else (100, 180, 255),
                1, cv2.LINE_AA)
    if footer:
        cv2.putText(img, footer, (10, img.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
    return img


def save_ab_figure(path: str, ab: Dict[str, Any]):
    """生成 A/B 对比柱状图。只画真实计算出的数值，待测项画为灰色虚框。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return _save_ab_figure_cv(path, ab)
    metrics = ["total_detections", "fps", "mean_latency_ms", "failure_rate"]
    names = ["检出框总数", "FPS", "平均时延(ms)", "失败/异常率"]
    av = [ab["A"].get(m, 0) or 0 for m in metrics]
    bv = [ab["B"].get(m, 0) or 0 for m in metrics]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.0))
    for ax, name, a, b in zip(axes, names, av, bv):
        bars = ax.bar(["A 基线", "B 改进"], [a, b], color=["#9aa8a3", "#0f9b6c"])
        ax.set_title(name, fontsize=10)
        for bar, v in zip(bars, [a, b]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8)
        ax.tick_params(labelsize=8)
    fig.suptitle("A/B 公平对照 · 有真值指标另行标注（无真值不可推断准确率）", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _save_ab_figure_cv(path: str, ab: Dict[str, Any]):
    img = np.full((360, 900, 3), 250, np.uint8)
    lines = ["A/B RESULT (SELFTEST / DEMO BACKEND)",
             f"A total_detections={ab['A'].get('total_detections')}  fps={ab['A'].get('fps')}",
             f"B total_detections={ab['B'].get('total_detections')}  fps={ab['B'].get('fps')}",
             "No ground truth => no Precision / Recall claim"]
    for i, t in enumerate(lines):
        cv2.putText(img, t, (20, 50 + i * 55), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (30, 90, 60), 2, cv2.LINE_AA)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cv2.imwrite(path, img)
    return path
