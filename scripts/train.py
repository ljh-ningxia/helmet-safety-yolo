"""训练脚本（正式后端）。

⚠️ 请勿用 data/synthetic/ 训练后对外声称精度 —— 合成数据只用于跑通流程。
真实训练前请按 docs/DATASET.md 准备 data/real/ 并固定 test/ 划分。

    python -m pip install ultralytics
    python scripts/train.py --data configs/data.yaml --epochs 100 --imgsz 640
"""

from __future__ import annotations

import argparse
import os
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "configs", "data.yaml"))
    ap.add_argument("--base", default="yolov8n.pt", help="官方预训练权重（需自行下载）")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--project", default=os.path.join(ROOT, "runs", "train"))
    ap.add_argument("--name", default="exp")
    ap.add_argument("--allow-synthetic", action="store_true",
                    help="仅用于流程自检；产出权重不得用于性能报告")
    args = ap.parse_args()

    with open(args.data, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    base = os.path.join(ROOT, cfg["path"])
    if "synthetic" in str(base) and not args.allow_synthetic:
        raise SystemExit("拒绝用合成数据训练正式权重。若只是验证流程，请加 --allow-synthetic，"
                         "且产出权重禁止用于性能报告。")

    # ultralytics 要求 data.yaml 中 path 为绝对路径
    data = dict(cfg)
    data.pop("note", None)
    data["path"] = base
    tmp = os.path.join(ROOT, "configs", "_generated_data.yaml")
    with open(tmp, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
    print(f"数据集配置已生成：{tmp}")

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("请先安装：python -m pip install ultralytics")

    model = YOLO(args.base)
    model.train(data=tmp, epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, device=args.device,
                project=args.project, name=args.name, exist_ok=True)
    print("训练完成。请把 best.pt 路径填入 run.py ab --weights，并在带真值 test/ 上重跑 A/B。")


if __name__ == "__main__":
    main()
