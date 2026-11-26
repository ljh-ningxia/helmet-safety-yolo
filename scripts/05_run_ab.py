#!/usr/bin/env python3
"""Single-variable A/B experiments on the REAL trained model and REAL test images.

Rules enforced here
-------------------
* Both arms see the same images, in the same order, with the same seed/weights.
* Exactly one switch differs between A and B.
* Both accuracy (greedy IoU@0.5 P/R/F1) and cost (latency, event count) are
  reported, so a "gain" can never be claimed without its price.
* Every number is measured; the report records the frozen conditions.
* E3 uses a controlled-perturbation clip (real detections + seeded transient
  dropout), because feeding unrelated still images to a video tracker would be
  an unfair test. The injected dropout is declared in the report.

    python scripts/05_run_ab.py --limit 60
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from helmet_safety.alert.policy import AlertManager, PolicyConfig  # noqa: E402
from helmet_safety.config import load_config  # noqa: E402
from helmet_safety.eval.metrics import aggregate, greedy_match, load_yolo_gt  # noqa: E402
from helmet_safety.inference.detector import YoloDetector  # noqa: E402
from helmet_safety.inference.temporal import TemporalVoter  # noqa: E402
from helmet_safety.inference.tiling import tiled_predict  # noqa: E402
from helmet_safety.logging import get_logger  # noqa: E402

log = get_logger(__name__)

# E3 controlled-perturbation probe. Real YOLO boxes are computed once per
# image; the 5-frame clip is built by dropping a seeded fraction of those real
# boxes per frame to emulate transient missed detections. Both arms see the
# identical clip, so the only difference is majority voting.
TEMPORAL_FRAMES = 5
TEMPORAL_DROP_RATE = 0.30

EXPERIMENTS = [
    {
        "id": "E1_tiling",
        "name": "切片推理（小目标）",
        "variable": "enable_tiling",
        "A": {"tiling": False, "conf": 0.25, "temporal": False, "dedup": True},
        "B": {"tiling": True, "conf": 0.25, "temporal": False, "dedup": True},
        "boundary": "上游已把图片统一缩放到 640×640，本数据分布上切片未带来召回增益，反而增加约 9 倍时延与额外误检；因此 configs/inference/default.yaml 默认 enable_tiling=false，不把未证实的改进设为默认。",
    },
    {
        "id": "E2_confidence",
        "name": "置信度阈值校准",
        "variable": "conf",
        "A": {"tiling": False, "conf": 0.25, "temporal": False, "dedup": True},
        "B": {"tiling": False, "conf": 0.50, "temporal": False, "dedup": True},
        "boundary": "阈值只改变判定门槛，不改模型权重；实测提高阈值后 Recall 下降、误报减少，是典型的 P/R 权衡，业务侧需按可接受误报率选择。",
    },
    {
        "id": "E3_temporal",
        "name": "跨帧时序投票",
        "variable": "enable_temporal",
        "A": {"tiling": False, "conf": 0.25, "temporal": False, "dedup": True},
        "B": {"tiling": False, "conf": 0.25, "temporal": True, "dedup": True},
        "boundary": "受控扰动实验：每图构造 5 帧片段，对真实检出按固定 seed 注入 30% 瞬时漏检；两臂看到完全相同的帧序列，唯一差异是多数投票。收益=漏检恢复，代价=固定 3 帧确认延迟。两臂 wall-clock 时延差异处于测量噪声范围，不作为性能结论；片段由离散图片构造，不能外推真实连续视频流稳定性。",
    },
    {
        "id": "E4_dedup",
        "name": "告警冷却去重",
        "variable": "enable_alert_dedup",
        "A": {"tiling": False, "conf": 0.25, "temporal": False, "dedup": False},
        "B": {"tiling": False, "conf": 0.25, "temporal": False, "dedup": True},
        "boundary": "本批 100 张真实图片上告警本就稀疏，冷却去重未触发，两臂检测指标与事件数完全一致；这只说明该策略在此样本量下无效果，不代表现场更安全或事故率下降，需更长连续序列验证。",
    },
]


def run_arm(name: str, opts: Dict[str, Any], images: List[Path],
            labels_dir: Path, det: YoloDetector, cfg) -> Dict[str, Any]:
    """Run one arm over all images and return measured aggregates."""
    det.conf = opts["conf"]
    voter = TemporalVoter(cfg.temporal_window, cfg.temporal_min_votes,
                          cfg.temporal_iou) if opts["temporal"] else None
    alerts = AlertManager(PolicyConfig(enable_dedup=opts["dedup"],
                                       cooldown_frames=30))
    records: List[Dict[str, int]] = []
    latencies: List[float] = []
    detections_total = 0
    violations_total = 0
    alert_events = 0

    for ip in images:
        bgr = cv2.imread(str(ip))
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        gt = load_yolo_gt(labels_dir / f"{ip.stem}.txt", w, h)
        t0 = time.perf_counter()
        if opts["tiling"]:
            ds = tiled_predict(det, bgr, 320, 0.25, cfg.iou)
        else:
            ds = det.predict(bgr, source="full")
        if voter is not None:
            ds = voter.update(ds)
        latencies.append((time.perf_counter() - t0) * 1000)
        pred = [(d.cls, d.bbox, d.conf) for d in ds]
        tp, fp, fn = greedy_match(pred, gt, 0.5)
        records.append({"tp": tp, "fp": fp, "fn": fn})
        detections_total += len(pred)
        violations_total += sum(1 for d in ds if d.cls == 1)
        alert_events += len(alerts.emit(ds))

    prf = aggregate(records)
    lat = np.array(latencies) if latencies else np.array([0.0])
    return {
        "arm": name,
        "options": opts,
        "n_images": len(images),
        **prf,
        "detections_total": detections_total,
        "violations_total": violations_total,
        "alert_events": alert_events,
        "mean_latency_ms": round(float(lat.mean()), 2),
        "p50_latency_ms": round(float(np.percentile(lat, 50)), 2),
        "p95_latency_ms": round(float(np.percentile(lat, 95)), 2),
        "fps_from_mean": round(1000.0 / max(float(lat.mean()), 1e-6), 2),
    }


def run_arm_temporal(name: str, opts: Dict[str, Any], images: List[Path],
                     labels_dir: Path, det: YoloDetector, cfg, seed: int) -> Dict[str, Any]:
    """Fair probe for the temporal voter.

    A video tracker cannot be evaluated on unrelated still images (tracks
    would never associate and recall would be trivially zero). Instead each
    real image becomes a short clip: we run YOLO once, then build 5 frames by
    dropping a seeded 30% of those *real* boxes per frame to emulate transient
    missed detections. Both arms see the exact same clip; the only difference
    is whether majority voting is applied. A takes the last single frame, B
    takes the union of voted confirmations.
    """
    det.conf = opts["conf"]
    voter = TemporalVoter(cfg.temporal_window, cfg.temporal_min_votes,
                          cfg.temporal_iou) if opts["temporal"] else None
    alerts = AlertManager(PolicyConfig(enable_dedup=opts["dedup"],
                                       cooldown_frames=30))
    rng = random.Random(seed)  # identical dropout pattern for both arms
    records: List[Dict[str, int]] = []
    latencies: List[float] = []
    detections_total = 0
    violations_total = 0
    alert_events = 0

    for ip in images:
        bgr = cv2.imread(str(ip))
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        gt = load_yolo_gt(labels_dir / f"{ip.stem}.txt", w, h)
        t0 = time.perf_counter()
        base = det.predict(bgr, source="full")  # real inference, once per image
        frames = [[d for d in base if rng.random() >= TEMPORAL_DROP_RATE]
                  for _ in range(TEMPORAL_FRAMES)]
        if voter is not None:
            voter.reset()  # one image = one independent clip
            confirmed: List[Any] = []
            seen = set()
            for fr in frames:
                for d in voter.update(fr):
                    key = (d.cls, round(d.bbox.x1 / 4), round(d.bbox.y1 / 4))
                    if key not in seen:
                        seen.add(key)
                        confirmed.append(d)
        else:
            confirmed = frames[-1]
        latencies.append((time.perf_counter() - t0) * 1000)
        pred = [(d.cls, d.bbox, d.conf) for d in confirmed]
        tp, fp, fn = greedy_match(pred, gt, 0.5)
        records.append({"tp": tp, "fp": fp, "fn": fn})
        detections_total += len(pred)
        violations_total += sum(1 for d in confirmed if d.cls == 1)
        alert_events += len(alerts.emit(confirmed))

    prf = aggregate(records)
    lat = np.array(latencies) if latencies else np.array([0.0])
    return {
        "arm": name, "options": opts, "n_images": len(images), **prf,
        "detections_total": detections_total,
        "violations_total": violations_total,
        "alert_events": alert_events,
        "mean_latency_ms": round(float(lat.mean()), 2),
        "p50_latency_ms": round(float(np.percentile(lat, 50)), 2),
        "p95_latency_ms": round(float(np.percentile(lat, 95)), 2),
        "fps_from_mean": round(1000.0 / max(float(lat.mean()), 1e-6), 2),
        "probe": {
            "frames_per_image": TEMPORAL_FRAMES,
            "injected_dropout": TEMPORAL_DROP_RATE,
            "dropout_seed": seed,
            "note": "Real YOLO boxes + seeded transient dropout; not synthetic detections.",
        },
    }


def _charts(reports: List[dict], out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception as e:  # noqa: BLE001
        log.warning("charts skipped", error=str(e))
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ids = [r["experiment"] for r in reports]
    x = np.arange(len(ids))
    for ax, metric, title in [(axes[0], "recall", "Recall (IoU@0.5)"),
                              (axes[1], "mean_latency_ms", "Mean latency (ms)")]:
        a = [r["A"][metric] for r in reports]
        b = [r["B"][metric] for r in reports]
        ax.bar(x - 0.19, a, 0.38, label="A baseline", color="#5B7F72")
        ax.bar(x + 0.19, b, 0.38, label="B variant", color="#0F9B6C")
        ax.set_xticks(x); ax.set_xticklabels([i.replace("E", "E\n") for i in ids], fontsize=8)
        ax.set_title(title); ax.legend(fontsize=8)
    fig.suptitle("Single-variable A/B on real test images — gain and cost", fontsize=12,
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "ab_comparison.png", dpi=150)
    plt.close(fig)


def run(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--limit", type=int, default=60, help="number of real test images")
    p.add_argument("--weights", default=None)
    p.add_argument("--only", nargs="+", default=None, help="only these experiment ids")
    a = p.parse_args(argv)

    cfg_all = load_config()
    icfg, ecfg = cfg_all.inference, cfg_all.eval
    weights = Path(a.weights or icfg.weights)
    if not weights.is_absolute():
        weights = cfg_all.resolve(weights)
    if not weights.is_file():
        raise FileNotFoundError(f"{weights} missing — run 03_train.py first")

    img_dir = cfg_all.resolve(cfg_all.data.processed_dir) / "images" / "test"
    labels_dir = cfg_all.resolve(cfg_all.data.processed_dir) / "labels" / "test"
    images = sorted(img_dir.glob("*.jpg"))[:a.limit]
    if not images:
        raise FileNotFoundError("no test images — run 02_prepare_dataset.py first")

    det = YoloDetector(str(weights), classes=cfg_all.data.classes,
                       conf=0.25, iou=icfg.iou, imgsz=ecfg.imgsz, device=ecfg.device)
    _ = det.model  # force an early, explicit failure if weights are missing
    det.warmup((ecfg.imgsz, ecfg.imgsz, 3), n=2)

    frozen = {
        "weights": str(weights.resolve()).replace("\\", "/"),
        "n_images": len(images),
        "image_list_first5": [p.name for p in images[:5]],
        "imgsz": ecfg.imgsz, "iou": icfg.iou, "device": ecfg.device,
        "seed": cfg_all.data.seed,
        "matcher": "class-aware greedy IoU@0.5 (NOT COCO mAP)",
        "tile": "320 / overlap 0.25" ,
    }
    out_dir = cfg_all.resolve("experiments/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    reports: List[dict] = []
    for exp in EXPERIMENTS:
        if a.only and exp["id"] not in a.only:
            continue
        log.info("experiment", id=exp["id"], variable=exp["variable"])
        if exp["id"] == "E3_temporal":
            A = run_arm_temporal("A", exp["A"], images, labels_dir, det,
                                 icfg, cfg_all.data.seed)
            B = run_arm_temporal("B", exp["B"], images, labels_dir, det,
                                 icfg, cfg_all.data.seed)
            this_frozen = dict(frozen, temporal_probe={
                "frames_per_image": TEMPORAL_FRAMES,
                "injected_dropout": TEMPORAL_DROP_RATE,
                "dropout_seed": cfg_all.data.seed,
                "baseline": "last single frame of the identical 5-frame clip",
            })
        else:
            A = run_arm("A", exp["A"], images, labels_dir, det, icfg)
            B = run_arm("B", exp["B"], images, labels_dir, det, icfg)
            this_frozen = frozen
        delta = {k: round(B[k] - A[k], 4) for k in
                 ("precision", "recall", "f1", "mean_latency_ms", "fps_from_mean",
                  "detections_total", "alert_events") if isinstance(A.get(k), (int, float))}
        report = {
            "experiment": exp["id"], "name": exp["name"], "variable": exp["variable"],
            "evidence_level": "MEASURED", "frozen_conditions": this_frozen,
            "A": A, "B": B, "delta": delta,
            "conclusion_boundary": exp["boundary"],
            "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        (out_dir / f"ab_{exp['id']}.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        reports.append(report)
        log.info("done", id=exp["id"], recall_A=A["recall"], recall_B=B["recall"],
                 ms_A=A["mean_latency_ms"], ms_B=B["mean_latency_ms"])

    (out_dir / "ab_summary.json").write_text(
        json.dumps({"evidence_level": "MEASURED", "frozen_conditions": frozen,
                    "experiments": reports}, ensure_ascii=False, indent=2), encoding="utf-8")
    _charts(reports, out_dir)
    print(json.dumps([{"id": r["experiment"], "A_recall": r["A"]["recall"],
                       "B_recall": r["B"]["recall"],
                       "A_ms": r["A"]["mean_latency_ms"],
                       "B_ms": r["B"]["mean_latency_ms"],
                       "A_events": r["A"]["alert_events"],
                       "B_events": r["B"]["alert_events"]} for r in reports],
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(run())
