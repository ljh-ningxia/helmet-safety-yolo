#!/usr/bin/env python3
"""统一入口 run.py —— 复现只需这一条命令。

    python run.py selftest                 # 端到端链路自检（不产生性能结论）
    python run.py web                      # 启动实时预警看板 http://127.0.0.1:5000
    python run.py ab                       # 三组公平 A/B 对照实验（demo 演练）
    python run.py ab --backend yolo --weights <best.pt> --source <测试集目录>
    python run.py detect --source <图片> --out <证据图路径>
    python run.py synthetic --count 60     # 导出合成数据集（仅流程自检）
    python run.py report                   # 由实验结果生成实验卡
"""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from experiments import run_ab as ab_module     # noqa: E402
from src import metrics, visualize             # noqa: E402
from src.datasource import FrameSource         # noqa: E402
from src.detector import build_detector, environment_fingerprint  # noqa: E402
from src.pipeline import PipelineConfig, SafetyPipeline           # noqa: E402
from src.synthetic import export_dataset       # noqa: E402


def cmd_selftest(args):
    print("=" * 68)
    print("端到端链路自检（SELFTEST_ONLY，不产生任何性能结论）")
    print("=" * 68)
    det = build_detector("demo")
    cfg = PipelineConfig(enable_tiling=True, enable_temporal=True,
                         enable_alert_dedup=True, save_alerts=False)
    pipe = SafetyPipeline(det, cfg)
    src = FrameSource("synthetic", frames=12, seed=20261126)
    frames = list(src)
    total_alerts = 0
    for i, (frame, gt) in enumerate(frames):
        r = pipe.process(frame)
        total_alerts += len(r.alerts)
        print(f"  帧 {i+1:02d}: 检测框 {len(r.detections):2d} / "
              f"确认违规 {len(r.confirmed):2d} / 新增告警 {len(r.alerts)} / "
              f"耗时 {r.latency_ms:6.1f} ms")
    assert all(len(f.shape) == 3 for f, _ in frames), "帧格式异常"
    assert len(pipe.results) == 12, "结果收集异常"
    # 指标模块自检
    preds = [r.detections for r in pipe.results]
    gts = [g for _, g in frames]
    ev = metrics.evaluate(preds, gts, [r.latency_ms for r in pipe.results])
    assert ev["ground_truth"] is True, "真值链路异常"
    no_gt = metrics.evaluate(preds, None, [1.0] * 12)
    assert no_gt["insufficient_ground_truth"] is True, "无真值保护失效"
    assert no_gt["precision"] is None, "无真值时不应给出 Precision"
    print("-" * 68)
    print(f"有真值链路：P={ev['precision']} R={ev['recall']} mAP50={ev['map50']} "
          f"（合成数据自检值，非模型性能）")
    print("无真值保护：正确拒绝输出 Precision / Recall / mAP ✓")
    print(f"环境指纹：{environment_fingerprint()['hash']}")
    print(f"告警事件累计：{total_alerts}（含分级与冷却去重）")
    print("✓ 自检通过。请继续运行 `python run.py ab` 与 `python run.py web`。")
    return 0


def cmd_detect(args):
    det_kw = {"conf": args.conf}
    if args.backend == "yolo":
        det_kw.update(weights=args.weights, iou=args.iou)
    else:
        det_kw.update(seed=args.seed)
    det = build_detector(args.backend, **det_kw)
    cfg = PipelineConfig(enable_tiling=args.tiling, enable_temporal=args.temporal,
                         enable_alert_dedup=True, save_alerts=True)
    pipe = SafetyPipeline(det, cfg)
    src = FrameSource(args.source, frames=args.frames, seed=args.seed)
    for i, (frame, _) in enumerate(src):
        r = pipe.process(frame)
        if i == 0 or args.save_all:
            img = visualize.draw_frame(frame, r.detections, r.confirmed,
                                       det.describe()["evidence_level"],
                                       footer=(f"Frame {r.frame_id} | det {len(r.detections)} | "
                                               f"violations {len(r.confirmed)} | {r.latency_ms:.0f} ms | "
                                               f"A baseline / B controlled by --tiling --temporal"))
            out = args.out if i == 0 else args.out.replace(".jpg", f"_{i:03d}.jpg")
            os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
            import cv2
            cv2.imwrite(out, img)
            print(f"证据图已保存：{out}")
    print(f"处理完成，共 {pipe.frame_id} 帧，告警 {len(pipe.alerts.events)} 条")
    return 0


def cmd_web(args):
    from src.webapp.app import create_app
    app = create_app(backend=args.backend, weights=args.weights, source=args.source)
    print(f"预警看板启动：http://127.0.0.1:{args.port}")
    print("提示：默认 demo 后端为演练器，页面顶部会持续显示证据等级水印。")
    app.run(host="127.0.0.1", port=args.port, debug=False, threaded=True)


def cmd_synthetic(args):
    out = export_dataset(os.path.join(ROOT, "data", "synthetic"),
                         count=args.count, seed=args.seed)
    print(f"合成数据集已导出：{out}")
    print("⚠️ 仅用于跑通训练/评估流程，不可作为性能证据。")
    return 0


def cmd_report(args):
    from scripts.make_report import build
    path = build(ROOT)
    print(f"实验卡已生成：{path}")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description="工地安全帽识别与预警系统")
    sub = p.add_subparsers(dest="cmd")

    s = sub.add_parser("selftest", help="端到端链路自检")
    s.set_defaults(func=cmd_selftest)

    ab_module.build_parser(sub)

    s = sub.add_parser("detect", help="对图片/视频生成带框证据图")
    s.add_argument("--backend", default="demo", choices=["demo", "yolo"])
    s.add_argument("--weights", default="runs/train/exp/weights/best.pt")
    s.add_argument("--source", default="synthetic")
    s.add_argument("--out", default="experiments/results/evidence.jpg")
    s.add_argument("--frames", type=int, default=1)
    s.add_argument("--seed", type=int, default=20261126)
    s.add_argument("--conf", type=float, default=0.25)
    s.add_argument("--iou", type=float, default=0.45)
    s.add_argument("--tiling", action="store_true")
    s.add_argument("--temporal", action="store_true")
    s.add_argument("--save-all", action="store_true")
    s.set_defaults(func=cmd_detect)

    s = sub.add_parser("web", help="启动实时预警看板")
    s.add_argument("--backend", default="demo", choices=["demo", "yolo"])
    s.add_argument("--weights", default="runs/train/exp/weights/best.pt")
    s.add_argument("--source", default="synthetic")
    s.add_argument("--port", type=int, default=5000)
    s.set_defaults(func=cmd_web)

    s = sub.add_parser("synthetic", help="导出合成数据集（仅流程自检）")
    s.add_argument("--count", type=int, default=60)
    s.add_argument("--seed", type=int, default=20261126)
    s.set_defaults(func=cmd_synthetic)

    s = sub.add_parser("report", help="由实验结果生成实验卡")
    s.set_defaults(func=cmd_report)

    ns = p.parse_args(argv)
    if not hasattr(ns, "func"):
        p.print_help()
        return 0
    return ns.func(ns) or 0


if __name__ == "__main__":
    sys.exit(main())
