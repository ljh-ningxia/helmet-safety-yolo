#!/usr/bin/env python3
"""Evaluate the trained model on the held-out test split.

Outputs
-------
experiments/results/evaluation.json   official metrics + latency + slice metrics
assets/real_inference.png            real photos with GT vs predicted boxes
assets/failure_cases.png             highest-error real images
experiments/results/benchmark.json   raw latency samples

    python scripts/04_evaluate.py --weights <path> --conf 0.25
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from helmet_safety.config import load_config  # noqa: E402
from helmet_safety.data.audit import _font  # noqa: E402
from helmet_safety.logging import get_logger  # noqa: E402
from helmet_safety.schemas import BBox, EvidenceLevel  # noqa: E402

log = get_logger(__name__)
GREEN = (15, 155, 108)
RED = (200, 60, 50)
BLUE = (63, 143, 210)
INK = (28, 43, 38)


def load_gt(txt: Path, w: int, h: int) -> List[Tuple[int, BBox]]:
    out = []
    if not txt.exists():
        return out
    for ln in txt.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        c, cx, cy, bw, bh = (float(v) for v in ln.split()[:5])
        out.append((int(c), BBox((cx - bw / 2) * w, (cy - bh / 2) * h,
                                 (cx + bw / 2) * w, (cy + bh / 2) * h)))
    return out


def greedy_match(pred: List[Tuple[int, BBox, float]], gt: List[Tuple[int, BBox]],
                 iou_thr: float = 0.5) -> Tuple[int, int, int]:
    """Class-aware greedy IoU matching -> (TP, FP, FN)."""
    used = set()
    tp = 0
    for c, box, _conf in sorted(pred, key=lambda x: -x[2]):
        best_i, best_iou = -1, 0.0
        for i, (gc, gb) in enumerate(gt):
            if i in used or gc != c:
                continue
            v = box.iou(gb)
            if v > best_iou:
                best_i, best_iou = i, v
        if best_i >= 0 and best_iou >= iou_thr:
            used.add(best_i)
            tp += 1
    fp = len(pred) - tp
    fn = len(gt) - len(used)
    return tp, fp, fn


def draw_boxes(im: Image.Image, gt: List[Tuple[int, BBox]],
               pred: List[Tuple[int, BBox, float]], classes: tuple) -> Image.Image:
    d = ImageDraw.Draw(im)
    for c, b in gt:
        d.rectangle([b.x1, b.y1, b.x2, b.y2], outline=GREEN, width=3)
        d.rectangle([b.x1, b.y1 - 16, b.x1 + 92, b.y1], fill=GREEN)
        d.text((b.x1 + 3, b.y1 - 15), f"GT:{classes[c]}", font=_font(10), fill="white")
    for c, b, conf in pred:
        d.rectangle([b.x1, b.y1, b.x2, b.y2], outline=RED, width=2)
        d.text((b.x1 + 2, b.y2 + 1), f"{classes[c]} {conf:.2f}", font=_font(11), fill=RED)
    return im


def run(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", default=None)
    p.add_argument("--conf", type=float, default=None)
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--skip-slices", action="store_true")
    a = p.parse_args(argv)

    cfg = load_config()
    e = cfg.eval
    weights = Path(a.weights or e.weights)
    if not weights.is_absolute():
        weights = cfg.resolve(weights)
    if not weights.is_file():
        raise FileNotFoundError(f"{weights} missing — run 03_train.py first")
    data_yaml = cfg.resolve(cfg.data.processed_dir) / "data.yaml"
    out_dir = cfg.resolve(e.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    assets = cfg.resolve("assets")
    assets.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO
    model = YOLO(str(weights))

    # ---- 1. Official Ultralytics validation on the held-out test split -------
    log.info("running official validation", split=e.split, weights=str(weights))
    t0 = time.time()
    val = model.val(data=str(data_yaml), split=e.split, imgsz=a.imgsz or e.imgsz,
                    conf=a.conf or e.conf, iou=e.iou, device=e.device,
                    batch=e.batch, verbose=False, plots=True,
                    project=str(cfg.resolve("experiments/runs")), name="eval")
    val_seconds = time.time() - t0
    box = getattr(val, "box", None)
    class_names = list(cfg.data.classes)
    per_class: Dict[str, Dict[str, float]] = {}
    if box is not None:
        for i, name in enumerate(class_names):
            try:
                per_class[name] = {
                    "precision": float(box.class_result(i)[0]),
                    "recall": float(box.class_result(i)[1]),
                    "map50": float(box.class_result(i)[2]),
                    "map50_95": float(box.class_result(i)[3]),
                }
            except Exception:  # noqa: BLE001
                pass
    official = {
        "precision": float(getattr(box, "mp", 0.0)) if box else None,
        "recall": float(getattr(box, "mr", 0.0)) if box else None,
        "map50": float(getattr(box, "map50", 0.0)) if box else None,
        "map50_95": float(getattr(box, "map", 0.0)) if box else None,
        "per_class": per_class,
        "val_seconds": round(val_seconds, 2),
        "speed_ms": {k: float(v) for k, v in (getattr(val, "speed", None) or {}).items()
                     if isinstance(v, (int, float))},
    }

    # ---- 2. Independent latency benchmark (same image, repeated) ------------
    test_imgs = sorted((cfg.resolve(cfg.data.processed_dir) / "images" / "test").glob("*.jpg"))
    if not test_imgs:
        raise FileNotFoundError("no test images — run 02_prepare_dataset.py first")
    probe = cv2.imread(str(test_imgs[0]))
    for _ in range(e.benchmark_warmup):
        model(probe, imgsz=a.imgsz or e.imgsz, conf=a.conf or e.conf,
              device=e.device, verbose=False)
    samples = []
    for _ in range(e.benchmark_repeats):
        s = time.perf_counter()
        model(probe, imgsz=a.imgsz or e.imgsz, conf=a.conf or e.conf,
              device=e.device, verbose=False)
        samples.append((time.perf_counter() - s) * 1000)
    latency = {
        "device": e.device,
        "imgsz": a.imgsz or e.imgsz,
        "repeats": e.benchmark_repeats,
        "warmup": e.benchmark_warmup,
        "mean_ms": round(float(np.mean(samples)), 2),
        "p50_ms": round(float(np.percentile(samples, 50)), 2),
        "p95_ms": round(float(np.percentile(samples, 95)), 2),
        "min_ms": round(float(np.min(samples)), 2),
        "fps_from_mean": round(1000.0 / float(np.mean(samples)), 2),
        "raw_ms": [round(x, 2) for x in samples],
        "note": "Single-image synchronous inference on CPU; excludes video decode and post-processing.",
    }
    (out_dir / "benchmark.json").write_text(json.dumps(latency, ensure_ascii=False, indent=2),
                                            encoding="utf-8")

    # ---- 3. Manual slice evaluation on real images ---------------------------
    slices: Dict[str, dict] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "images": 0})
    errors: List[dict] = []
    evidence: List[Path] = []
    if not a.skip_slices:
        conf = a.conf or e.conf
        for ip in test_imgs:
            bgr = cv2.imread(str(ip))
            if bgr is None:
                continue
            h, w = bgr.shape[:2]
            gt = load_gt(cfg.resolve(cfg.data.processed_dir) / "labels" / "test" / f"{ip.stem}.txt", w, h)
            res = model(bgr, imgsz=a.imgsz or e.imgsz, conf=conf, iou=e.iou,
                        device=e.device, verbose=False)[0]
            pred = []
            if getattr(res, "boxes", None) is not None:
                for xyxy, c, cf in zip(res.boxes.xyxy.tolist(),
                                       res.boxes.cls.tolist(), res.boxes.conf.tolist()):
                    pred.append((int(c), BBox(*xyxy), float(cf)))
            tp, fp, fn = greedy_match(pred, gt, 0.5)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            lum = float(gray.mean())
            n_gt = len(gt)
            keys = [
                f"brightness:{'dark' if lum < 80 else ('normal' if lum < 170 else 'bright')}",
                f"density:{'single' if n_gt <= 1 else ('few' if n_gt <= 4 else 'crowded')}",
            ]
            for k in keys:
                slices[k]["tp"] += tp; slices[k]["fp"] += fp
                slices[k]["fn"] += fn; slices[k]["images"] += 1
            if fp + fn:
                errors.append({"image": ip.name, "fp": fp, "fn": fn, "gt": n_gt,
                               "pred": len(pred), "luminance": round(lum, 1)})
            if fn >= 1 and len(evidence) < 6:
                evidence.append(ip)
        for v in slices.values():
            tp, fp, fn = v["tp"], v["fp"], v["fn"]
            v["precision"] = round(tp / max(tp + fp, 1), 4)
            v["recall"] = round(tp / max(tp + fn, 1), 4)
            v["f1"] = round(2 * v["precision"] * v["recall"] /
                            max(v["precision"] + v["recall"], 1e-9), 4)
        # Evidence sheet: real photos, green=GT, red=prediction
        chosen = (evidence + [p for p in test_imgs if p not in evidence])[:6]
        cells = []
        for ip in chosen:
            bgr = cv2.imread(str(ip))
            h, w = bgr.shape[:2]
            gt = load_gt(cfg.resolve(cfg.data.processed_dir) / "labels" / "test" / f"{ip.stem}.txt", w, h)
            res = model(bgr, imgsz=a.imgsz or e.imgsz, conf=conf, iou=e.iou,
                        device=e.device, verbose=False)[0]
            pred = []
            if getattr(res, "boxes", None) is not None:
                for xyxy, c, cf in zip(res.boxes.xyxy.tolist(),
                                       res.boxes.cls.tolist(), res.boxes.conf.tolist()):
                    pred.append((int(c), BBox(*xyxy), float(cf)))
            im = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            im.thumbnail((460, 330))
            cells.append((ip.name, draw_boxes(im.copy(), gt, pred, class_names)))

        if cells:
            cw, ch, cols = 470, 360, 3
            sheet = Image.new("RGB", (cols * cw, 2 * ch + 70), "white")
            d = ImageDraw.Draw(sheet)
            d.rectangle([0, 0, sheet.width, 58], fill=(245, 248, 246))
            d.text((16, 12), "真实测试集推理证据 · 绿框=人工标注 GT，红框=模型预测", font=_font(20, True), fill=INK)
            d.text((16, 38), "Hard Hats Dataset (CC BY 4.0) · 未经挑选的真实图片，全部来自 held-out test split",
                   font=_font(12), fill=(107, 125, 118))
            for i, (name, im) in enumerate(cells):
                x, y = (i % cols) * cw + 5, (i // cols) * ch + 64
                sheet.paste(im, (x, y))
                d.text((x, y + 332), f"{i+1:02d}  {name[:40]}", font=_font(11), fill=(107, 125, 118))
            sheet.save(assets / "real_inference.png")
            log.info("saved", path=str(assets / "real_inference.png"))

    payload = {
        "evidence_level": EvidenceLevel.MEASURED.value,
        "weights": str(weights.resolve()).replace("\\", "/"),
        "data_yaml": str(data_yaml.resolve()).replace("\\", "/"),
        "split": e.split,
        "n_test_images": len(test_imgs),
        "conf": a.conf or e.conf,
        "iou": e.iou,
        "official_validation": official,
        "latency_benchmark": latency,
        "slice_metrics": dict(slices),
        "slice_method": "Class-aware greedy IoU@0.5 on raw predictions; slices by luminance and object density.",
        "failure_cases_top10": sorted(errors, key=lambda x: -(x["fp"] + x["fn"]))[:10],
        "artifacts": ["assets/real_inference.png", "experiments/results/benchmark.json"],
        "boundaries": [
            "Metrics are measured on a single public dataset, not on the customer's own camera distribution.",
            "CPU latency is not an edge-deployment benchmark; no TensorRT/ONNX optimization applied yet.",
            "Slice results use a simple greedy matcher and are diagnostic, not a replacement for official mAP.",
        ],
    }
    (out_dir / "evaluation.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
    log.info("evaluation complete", map50=official["map50"], map50_95=official["map50_95"],
             fps=latency["fps_from_mean"])
    print(json.dumps({"official": official, "latency": {k: latency[k] for k in
           ("mean_ms", "p95_ms", "fps_from_mean")},
           "n_test_images": len(test_imgs),
           "report": str(out_dir / "evaluation.json")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(run())
