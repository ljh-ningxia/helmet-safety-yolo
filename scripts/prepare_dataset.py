"""数据集接入校验：确认目录、标签、类别与测试集划分，避免「数据没对齐就开跑」。

    python scripts/prepare_dataset.py --root data/real
"""

from __future__ import annotations

import argparse
import os
from collections import Counter


def check(root: str) -> int:
    problems = []
    for split in ("train", "val", "test"):
        img = os.path.join(root, "images", split)
        lab = os.path.join(root, "labels", split)
        if not os.path.isdir(img):
            problems.append(f"缺少目录 {img}")
            continue
        files = [f for f in os.listdir(img) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        print(f"[{split}] 图片 {len(files)} 张")
        if not files:
            problems.append(f"{split} 为空")
        counter = Counter()
        missing = 0
        for f in files:
            lp = os.path.join(lab, os.path.splitext(f)[0] + ".txt")
            if not os.path.exists(lp):
                missing += 1
                continue
            with open(lp, encoding="utf-8") as fh:
                for line in fh:
                    p = line.split()
                    if len(p) == 5:
                        counter[int(p[0])] += 1
        print(f"       标签缺失 {missing} 个；类别分布 {dict(counter)}")
        if split == "test" and missing:
            problems.append("test 集缺少真值标签 → 无法计算 P/R/mAP")
        if any(k not in (0, 1) for k in counter):
            problems.append(f"{split} 存在非 0/1 类别，请按 helmet=0 / no_helmet=1 映射")

    print("-" * 60)
    if problems:
        print("校验未通过：")
        for p in problems:
            print("  ✗", p)
        return 1
    print("✓ 校验通过。请记录 test/ 的帧数与划分方式到实验卡，之后不要再改动。")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/real")
    raise SystemExit(check(ap.parse_args().root))
