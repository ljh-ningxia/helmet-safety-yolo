"""REST API for the helmet-safety service.

Design: the service starts even when a weight file is absent (health checks and
static dashboard still work), but inference endpoints fail loudly with 503
instead of silently falling back to a fake detector.
"""
from __future__ import annotations

import base64
import json
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..config import ROOT, load_config
from ..logging import get_logger
from ..schemas import BBox, EvidenceLevel

log = get_logger(__name__)
cfg = load_config()
RESULTS = cfg.resolve("experiments/results")
ASSETS = cfg.resolve("assets")
WEB = ROOT / "web"

app = FastAPI(title="Helmet Safety API", version="2.0.0",
              description="YOLO-based construction-site hard-hat detection and alerting")

_state: Dict[str, Any] = {"pipeline": None, "load_error": ""}


class DetectionOut(BaseModel):
    bbox: List[float]
    conf: float
    cls: int
    label: str
    track_id: Optional[int] = None


class DetectResponse(BaseModel):
    frame_id: int
    latency_ms: float
    n_detections: int
    n_violations: int
    detections: List[DetectionOut]
    alerts: List[dict]
    evidence_level: str
    annotated_image_base64: str = Field(default="", description="JPEG with drawn boxes")


def _read_json(name: str) -> dict:
    p = RESULTS / name
    if not p.is_file():
        raise HTTPException(404, f"{name} not found — run the corresponding pipeline step")
    return json.loads(p.read_text(encoding="utf-8"))


def get_pipeline():
    if _state["pipeline"] is None:
        from ..alert.policy import JsonlSink, LogSink, PolicyConfig
        from ..inference.detector import YoloDetector
        from ..inference.pipeline import SafetyPipeline
        weights = cfg.resolve(cfg.inference.weights)
        if not weights.is_file():
            raise HTTPException(503, f"weights missing: {weights}. Train first (scripts/03_train.py).")
        det = YoloDetector(str(weights), classes=cfg.data.classes,
                           conf=cfg.inference.conf, iou=cfg.inference.iou,
                           imgsz=cfg.inference.imgsz, device=cfg.inference.device)
        acfg = PolicyConfig(enable_dedup=cfg.alert.enable_dedup,
                            cooldown_frames=cfg.alert.cooldown_frames,
                            level_2_count=cfg.alert.level_2_count,
                            level_3_count=cfg.alert.level_3_count)
        sinks = [LogSink(), JsonlSink(RESULTS / "alerts.jsonl")]
        _state["pipeline"] = SafetyPipeline(det, cfg.inference, acfg, sinks)
        log.info("pipeline ready", weights=str(weights))
    return _state["pipeline"]


def _annotate(bgr: np.ndarray, result) -> str:
    for d in result.detections:
        b: BBox = d.bbox
        color = (15, 155, 108) if d.cls == 0 else (60, 60, 200)
        cv2.rectangle(bgr, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), color, 2)
        cv2.putText(bgr, f"{d.label} {d.conf:.2f}", (int(b.x1), max(14, int(b.y1) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")


@app.get("/health")
def health() -> dict:
    w = cfg.resolve(cfg.inference.weights)
    return {"status": "ok", "version": "2.0.0", "env": cfg.env,
            "model_loaded": _state["pipeline"] is not None,
            "weights_available": w.is_file(),
            "device": cfg.inference.device}


@app.get("/api/v1/config")
def config() -> dict:
    return cfg.to_dict()


@app.get("/api/v1/metrics")
def metrics() -> dict:
    return _read_json("evaluation.json")


@app.get("/api/v1/dataset")
def dataset() -> dict:
    return _read_json("dataset_audit.json")


@app.get("/api/v1/ab")
def ab() -> dict:
    return _read_json("ab_summary.json")


@app.get("/api/v1/samples")
def samples() -> dict:
    p = _read_json("dataset_audit.json")
    return {"contact_sheet": p.get("contact_sheet", ""),
            "real_inference": "assets/real_inference.png"}


@app.post("/api/v1/detect", response_model=DetectResponse)
async def detect(file: UploadFile = File(...)) -> DetectResponse:
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(400, "an image file is required")
    raw = await file.read()
    if len(raw) > 20 * 1024 * 1024:
        raise HTTPException(413, "image too large (max 20 MiB)")
    arr = np.frombuffer(raw, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(400, "cannot decode image")
    pipe = get_pipeline()
    t0 = time.perf_counter()
    result = pipe.process(bgr)
    image_b64 = _annotate(bgr.copy(), result)
    return DetectResponse(
        frame_id=result.frame_id,
        latency_ms=round((time.perf_counter() - t0) * 1000, 2),
        n_detections=len(result.detections),
        n_violations=sum(1 for d in result.confirmed if d.cls == 1),
        detections=[DetectionOut(bbox=[d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2],
                                 conf=d.conf, cls=d.cls, label=d.label,
                                 track_id=d.track_id) for d in result.detections],
        alerts=[a.to_dict() for a in result.alerts],
        evidence_level=EvidenceLevel.MEASURED.value,
        annotated_image_base64=image_b64,
    )


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    f = WEB / "index.html"
    if not f.is_file():
        return HTMLResponse("<h1>Helmet Safety API</h1><p>Dashboard missing.</p>")
    return HTMLResponse(f.read_text(encoding="utf-8"))


if ASSETS.is_dir():
    app.mount("/assets", StaticFiles(directory=str(ASSETS)), name="assets")
