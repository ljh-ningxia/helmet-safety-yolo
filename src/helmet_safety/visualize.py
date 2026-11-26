"""Drawing helpers for CLI evidence output (OpenCV, no external font download)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .schemas import FrameResult

GREEN = (15, 155, 108)
RED = (60, 60, 200)


def draw_result(bgr: np.ndarray, result: FrameResult, out: str | Path) -> Path:
    canvas = bgr.copy()
    for d in result.detections:
        b = d.bbox
        color = GREEN if d.cls == 0 else RED
        cv2.rectangle(canvas, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), color, 2)
        cv2.putText(canvas, f"{d.label} {d.conf:.2f}",
                    (int(b.x1), max(15, int(b.y1) - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    cv2.putText(canvas,
                f"det={len(result.detections)} viol={sum(1 for x in result.confirmed if x.cls == 1)} "
                f"{result.latency_ms:.0f}ms  {result.evidence_level.value}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 3, cv2.LINE_AA)
    cv2.putText(canvas,
                f"det={len(result.detections)} viol={sum(1 for x in result.confirmed if x.cls == 1)} "
                f"{result.latency_ms:.0f}ms  {result.evidence_level.value}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1, cv2.LINE_AA)
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(target), canvas):
        raise IOError(f"failed to write {target}")
    return target
