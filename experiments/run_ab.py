"""公平 A/B 对照实验（对应课程规范：唯一主要变动 / 冻结条件 / 结论边界）。

三组独立实验，每组只改一项、其余全部冻结：
    E1  基线整图推理        → 重叠切片推理        （改动：定位/检测）
    E2  单帧即判            → 连续 K 帧投票确认   （改动：判定）
    E3  逐帧上报            → 分级 + 冷却去重     （改动：业务动作）

冻结项：同一帧序列、同一检测器与阈值、同一 NMS IoU、同一硬件与库版本、
        同一次运行内完成 A 与 B，A/B 由同一个 PipelineConfig 派生。

运行：
    python run.py ab --backend demo --frames 90
    python run.py ab --backend yolo --weights runs/train/exp/weights/best.pt \
                     --source data/real/images --frames 300
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import metrics
from src.datasource import FrameSource
from src.detector import build_detector, environment_fingerprint
from src.pipeline import PipelineConfig, SafetyPipeline
from src.visualize import save_ab_figure

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "experiments", "results")

EXPERIMENTS = [
    {
        "id": "E1_tiling",
        "variant": "B_tiling",
        "title": "远景小目标：整图推理 vs 重叠切片推理",
        "single_change": "仅把定位/检测环节从整图一次推理改为重叠切片 + 放大 + 跨切片 NMS",
        "metric_layer": "检测层",
        "boundary": "切片会增加推理次数与显存/CPU 开销，必须同时报告 FPS 与时延；小目标召回改善需人工真值支持，不能仅凭检出框变多推断。",
    },
    {
        "id": "E2_temporal",
        "variant": "B_temporal",
        "title": "告警稳定性：单帧即判 vs 连续 K 帧投票",
        "single_change": "仅把判定环节从单帧即报改为连续 3 帧命中确认",
        "metric_layer": "判定层",
        "boundary": "投票引入 K-1 帧固有延迟，快速通过画面的目标可能漏报；需报告确认延迟与漏报观察值。",
    },
    {
        "id": "E3_dedup",
        "variant": "B_dedup",
        "title": "事件闭环：逐帧上报 vs 分级 + 冷却去重",
        "single_change": "仅在业务动作环节加入分级、8 秒冷却去重与证据快照留存",
        "metric_layer": "业务层",
        "boundary": "去重会压低事件条数，不代表违规减少；冷却窗口内同一目标不再重复上报，需说明窗口设置依据。",
    },
]


def _clean(obj):
    """把本机绝对路径替换成仓库相对路径，保证 JSON 在别人机器上可读。"""
    if isinstance(obj, str):
        return obj.replace(ROOT.replace("\\", "/"), ".").replace(ROOT, ".").replace("\\", "/")
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    return obj


def _run_one(pipeline: SafetyPipeline, frames: List, gts: List) -> Dict[str, Any]:
    pipeline.reset()
    preds, lat, confirmed_counts, events = [], [], [], 0
    failures = 0
    for frame, gt in frames:
        try:
            r = pipeline.process(frame, emit=True)
        except Exception:
            failures += 1
            preds.append([])
            lat.append(0.0)
            confirmed_counts.append(0)
            continue
        preds.append(r.detections)
        lat.append(r.latency_ms)
        confirmed_counts.append(len(r.confirmed))
        events += len(r.alerts)
    out = metrics.evaluate(preds, gts, lat, failures)
    out["confirmed_total"] = sum(confirmed_counts)
    out["alert_events"] = events
    out["alert_summary"] = pipeline.alerts.summary()
    return out


def run(args):
    os.makedirs(RESULTS, exist_ok=True)
    src = FrameSource(args.source, frames=args.frames, seed=args.seed,
                      labels_dir=args.labels_dir)
    frames = list(src)
    gts = [g for _, g in frames] if all(g is not None for _, g in frames) else None
    if gts is None:
        print("[提示] 当前数据源无人工真值：本次只输出检测输出与开销统计，"
              "不输出 Precision / Recall / mAP。")

    det_kw = {"conf": args.conf}
    if args.backend == "yolo":
        det_kw.update(weights=args.weights, iou=args.iou, imgsz=args.imgsz)
    else:
        det_kw.update(min_area=args.min_area, seed=args.seed)
    detector = build_detector(args.backend, **det_kw)

    base_cfg = PipelineConfig(conf=args.conf, iou=args.iou, imgsz=args.imgsz,
                              save_alerts=False, log_dir=RESULTS)
    env = environment_fingerprint()
    level = detector.describe()["evidence_level"]
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    reports = []

    for exp in EXPERIMENTS:
        pa = SafetyPipeline(detector, base_cfg.variant("A"), collect=False)
        pb = SafetyPipeline(detector, base_cfg.variant(exp["variant"]), collect=False)
        A = _run_one(pa, frames, gts)
        B = _run_one(pb, frames, gts)
        delta = metrics.compare(A, B)
        report = _clean({
            "experiment": exp["id"],
            "title": exp["title"],
            "single_change": exp["single_change"],
            "metric_layer": exp["metric_layer"],
            "timestamp": stamp,
            "evidence_level": level,
            "warning": (
                "SELFTEST_ONLY：demo 后端为 HSV 颜色演练器，以下数值仅证明链路可运行、"
                "A/B 框架可复核，禁止作为 YOLO 模型性能或项目成果对外引用。"
                if level != "MEASURED" else
                "指标仅在本次冻结条件与给定测试集下成立，跨数据集/跨硬件不可外推。"
            ),
            "frozen_conditions": {
                "source": src.describe(),
                "detector": detector.describe(),
                "base_config": base_cfg.__dict__,
                "environment": env,
                "rule": "A/B 由同一配置派生，除 single_change 外无任何差异",
            },
            "A": A, "B": B, "delta": delta,
            "conclusion_boundary": exp["boundary"],
        })
        path = os.path.join(RESULTS, f"{exp['id']}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        fig = os.path.join(RESULTS, f"{exp['id']}.png")
        try:
            save_ab_figure(fig, {"A": A, "B": B})
        except Exception as e:
            print("图生成失败（不影响实验数据）:", e)
        reports.append(report)
        print(f"[{exp['id']}] A 检出={A['total_detections']}  FPS={A['fps']}  "
              f"事件={A['alert_events']} | B 检出={B['total_detections']}  "
              f"FPS={B['fps']}  事件={B['alert_events']}  → {path}")

    summary = {
        "generated_at": stamp, "evidence_level": level,
        "experiments": [r["experiment"] for r in reports],
        "ground_truth_available": gts is not None,
        "environment": env,
        "note": "本文件由 run.py ab 自动生成；请人工核对后再引用任何数字。",
    }
    with open(os.path.join(RESULTS, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n实验数据目录：{RESULTS}")
    print("请运行 `python run.py report` 生成实验卡；引用数字前请人工复核。")
    return 0


def build_parser(sub):
    p = sub.add_parser("ab", help="运行三组公平 A/B 对照实验")
    p.add_argument("--backend", default="demo", choices=["demo", "yolo"])
    p.add_argument("--weights", default="runs/train/exp/weights/best.pt")
    p.add_argument("--source", default="synthetic")
    p.add_argument("--labels-dir", default=None)
    p.add_argument("--frames", type=int, default=60)
    p.add_argument("--seed", type=int, default=20261126)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--min-area", type=int, default=300)
    p.set_defaults(func=run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")
    build_parser(sub)
    ns = parser.parse_args()
    if not hasattr(ns, "func"):
        parser.print_help()
    else:
        ns.func(ns)
