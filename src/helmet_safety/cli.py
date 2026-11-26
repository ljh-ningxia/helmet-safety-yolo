"""Unified command-line entry point.

    python -m helmet_safety.cli doctor
    python -m helmet_safety.cli info
    python -m helmet_safety.cli detect path/to/photo.jpg --out result.jpg
    python -m helmet_safety.cli serve --port 8000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_config
from .logging import configure, get_logger

log = get_logger(__name__)


def cmd_doctor(args) -> int:
    cfg = load_config()
    checks = []
    def chk(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    try:
        import torch
        import ultralytics  # noqa: F401
        chk("torch+ultralytics", True, f"torch {torch.__version__}, ultralytics {ultralytics.__version__}")
    except Exception as e:  # noqa: BLE001
        chk("torch+ultralytics", False, str(e))
    w = cfg.resolve(cfg.inference.weights)
    chk("trained_weights", w.is_file(), str(w))
    pw = cfg.resolve("models/yolov8n.pt")
    chk("pretrained_weights", pw.is_file(), str(pw))
    dy = cfg.resolve(cfg.data.processed_dir) / "data.yaml"
    chk("dataset_yaml", dy.is_file(), str(dy))
    for name in ("dataset_audit.json", "train_run.json", "evaluation.json", "ab_summary.json"):
        p = cfg.resolve("experiments/results") / name
        chk(name, p.is_file(), str(p))
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    failed = [c for c in checks if not c["ok"]]
    if failed:
        log.warning("doctor found missing artifacts", count=len(failed))
    return 0 if not failed else 1


def cmd_info(args) -> int:
    print(json.dumps(load_config().to_dict(), ensure_ascii=False, indent=2))
    return 0


def cmd_detect(args) -> int:
    import cv2

    from .alert.policy import LogSink, PolicyConfig
    from .inference.detector import YoloDetector
    from .inference.pipeline import SafetyPipeline
    cfg = load_config()
    src = Path(args.image)
    if not src.is_file():
        raise FileNotFoundError(src)
    bgr = cv2.imread(str(src))
    if bgr is None:
        raise ValueError(f"cannot decode {src}")
    det = YoloDetector(str(cfg.resolve(cfg.inference.weights)), classes=cfg.data.classes,
                       conf=cfg.inference.conf, iou=cfg.inference.iou,
                       imgsz=cfg.inference.imgsz, device=cfg.inference.device)
    pipe = SafetyPipeline(det, cfg.inference, PolicyConfig(), [LogSink()])
    r = pipe.process(bgr)
    out = {"file": str(src), **r.to_dict()}
    if args.out:
        from .visualize import draw_result
        draw_result(bgr, r, args.out)
        out["annotated_image"] = args.out
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


def cmd_serve(args) -> int:
    import uvicorn
    uvicorn.run("helmet_safety.api.app:app", host=args.host, port=args.port,
                reload=args.reload, log_level=args.log_level)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="helmet-safety", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log-level", default="INFO")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor", help="check environment, weights and artifacts")
    sub.add_parser("info", help="print resolved configuration")
    d = sub.add_parser("detect", help="run detection on one image")
    d.add_argument("image")
    d.add_argument("--out", default=None)
    s = sub.add_parser("serve", help="start the REST API + dashboard")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--reload", action="store_true")
    s.add_argument("--log-level", default="info")
    a = p.parse_args(argv)
    configure(a.log_level)
    return {"doctor": cmd_doctor, "info": cmd_info,
            "detect": cmd_detect, "serve": cmd_serve}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
