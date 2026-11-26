"""预警看板 Flask 应用。

页面顶部常驻「证据等级」声明：demo 后端下任何数字都不得被截图当成性能证据。
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict
from typing import Dict, Any

import cv2
from flask import Flask, Response, jsonify, render_template, request

from .. import visualize
from ..datasource import FrameSource
from ..detector import build_detector, environment_fingerprint
from ..pipeline import PipelineConfig, SafetyPipeline

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class State:
    def __init__(self, backend: str, weights: str, source: str):
        self.backend = backend
        self.weights = weights
        self.source = source
        self.lock = threading.Lock()
        self.frame_jpeg = b""
        self.events = []
        self.metrics = {}
        self.frame_id = 0
        self.cfg = PipelineConfig(save_alerts=True,
                                  log_dir=os.path.join(ROOT, "experiments", "results"))
        self.detector = build_detector(
            backend, **({"weights": weights, "conf": 0.25, "iou": 0.45}
                        if backend == "yolo" else {"conf": 0.25}))
        self.evidence_level = self.detector.describe()["evidence_level"]
        self.pipeline = self._new_pipeline()
        self.running = True

    def _new_pipeline(self):
        return SafetyPipeline(self.detector, self.cfg, collect=False)

    def set_switches(self, tiling: bool, temporal: bool, dedup: bool):
        with self.lock:
            self.cfg.enable_tiling = tiling
            self.cfg.enable_temporal = temporal
            self.cfg.enable_alert_dedup = dedup
            self.pipeline = self._new_pipeline()

    def loop(self):
        src = FrameSource(self.source, frames=10_000_000)
        try:
            for frame, _ in src:
                if not self.running:
                    break
                with self.lock:
                    r = self.pipeline.process(frame, emit=True)
                    img = visualize.draw_frame(
                        frame, r.detections, r.confirmed, self.evidence_level,
                        footer=(f"Frame {r.frame_id} | det {len(r.detections)} | "
                                f"violations {len(r.confirmed)} | {r.latency_ms:.0f} ms | "
                                f"tiling={self.cfg.enable_tiling} "
                                f"temporal={self.cfg.enable_temporal} "
                                f"dedup={self.cfg.enable_alert_dedup}"))
                    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
                    if ok:
                        self.frame_jpeg = buf.tobytes()
                    self.frame_id = r.frame_id
                    self.events = [e.to_dict() for e in self.pipeline.alerts.events[-30:]][::-1]
                    self.metrics = {
                        "frames": r.frame_id,
                        "detections": len(r.detections),
                        "confirmed": len(r.confirmed),
                        "latency_ms": round(r.latency_ms, 1),
                        "fps": round(1000.0 / r.latency_ms, 1) if r.latency_ms > 0 else 0,
                        "events": len(self.pipeline.alerts.events),
                        **self.pipeline.alerts.summary(),
                    }
                time.sleep(0.04)
        except Exception as e:
            with self.lock:
                self.metrics["error"] = str(e)

    def snapshot(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "metrics": dict(self.metrics),
                "events": list(self.events),
                "config": asdict(self.cfg),
                "changed_modules": self.cfg.changed_modules(),
                "evidence_level": self.evidence_level,
                "detector": self.detector.describe(),
                "environment": environment_fingerprint(),
            }


def create_app(backend: str = "demo", weights: str = "",
               source: str = "synthetic") -> Flask:
    here = os.path.dirname(os.path.abspath(__file__))
    app = Flask(__name__, template_folder=os.path.join(here, "templates"),
                static_folder=os.path.join(here, "static"))
    st = State(backend, weights, source)
    threading.Thread(target=st.loop, daemon=True).start()

    @app.route("/")
    def index():
        return render_template("index.html", evidence_level=st.evidence_level)

    @app.route("/video")
    def video():
        def gen():
            while True:
                with st.lock:
                    jpg = st.frame_jpeg
                if jpg:
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
                time.sleep(0.05)
        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/api/state")
    def state():
        return jsonify(st.snapshot())

    @app.route("/api/switch", methods=["POST"])
    def switch():
        d = request.get_json(force=True) or {}
        st.set_switches(bool(d.get("tiling")), bool(d.get("temporal")),
                        bool(d.get("dedup")))
        return jsonify({"ok": True, "changed_modules": st.cfg.changed_modules()})

    @app.route("/api/ab")
    def ab():
        out = []
        res = os.path.join(ROOT, "experiments", "results")
        for name in ("E1_tiling", "E2_temporal", "E3_dedup"):
            p = os.path.join(res, name + ".json")
            if os.path.exists(p):
                with open(p, encoding="utf-8") as f:
                    out.append(json.load(f))
            else:
                out.append({"experiment": name, "status": "待测"})
        return jsonify({"experiments": out, "evidence_level": st.evidence_level})

    @app.route("/api/evidence")
    def evidence():
        return jsonify({
            "five_questions": {
                "解决什么真实问题": "高位俯拍远景小目标漏检 + 单帧误检导致告警风暴 + 检测结果未形成处置闭环",
                "相对基线改了什么": "推理组织方式 / 跨帧判定 / 事件闭环，三项分别做单变量 A/B",
                "哪些实测数据支持": "experiments/results/*.json（含环境指纹）；无真值项明确标注待测",
                "本人负责哪些工作": "见 docs/CONTRIBUTION.md",
                "陌生人怎样复现": "README 四段式 + requirements.txt + run.py selftest / ab / web",
            },
            "environment": environment_fingerprint(),
        })

    return app
