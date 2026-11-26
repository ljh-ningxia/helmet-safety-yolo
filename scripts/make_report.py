"""由 experiments/results/*.json 生成实验卡与证据索引。

原则：数字只能来自真实运行产物；没有运行就写「待测」，绝不补写。
"""

from __future__ import annotations

import json
import os
from datetime import datetime

IDS = ["E1_tiling", "E2_temporal", "E3_dedup"]


def build(root: str) -> str:
    results = os.path.join(root, "experiments", "results")
    out = os.path.join(root, "docs", "EXPERIMENT_CARD.md")
    lines = [
        "# 实验卡 · 工地现场安全帽识别与预警系统",
        "",
        f"> 本文件由 `python run.py report` 于 {datetime.now():%Y-%m-%d %H:%M} 自动生成。",
        "> 所有数字来自 `experiments/results/` 的真实运行产物；未运行的项目一律标注「待测」。",
        "> **结论—证据—边界** 三段必须同时出现，缺一段视为无效结论。",
        "",
        "## 冻结条件（A / B 完全一致）",
        "",
        "| 设置项 | 冻结值 |",
        "| --- | --- |",
        "| 数据 | 同一测试集 / 同一帧序列（固定 seed），固定划分 |",
        "| 配置 | 同一推理流程、置信度阈值、NMS IoU |",
        "| 设备 | 同一硬件、同一 Python / OpenCV / 推理库版本 |",
        "| 指标 | 准确性（有真值时）、时延 / FPS、失败 / 异常率 |",
        "| 版本 | A、B 基于同一代码基线，**仅引入一项主要改动** |",
        "",
        "## 三组实验",
        "",
    ]
    found = 0
    for eid in IDS:
        path = os.path.join(results, f"{eid}.json")
        if not os.path.exists(path):
            lines += [
                f"### {eid} · 待测",
                "",
                "- 状态：**未运行**。运行 `python run.py ab` 后本段自动填充。",
                "- 结论：待测　证据：待测　边界：待测",
                "",
            ]
            continue
        found += 1
        with open(path, encoding="utf-8") as f:
            r = json.load(f)
        A, B = r["A"], r["B"]
        lines += [
            f"### {r['experiment']} · {r['title']}",
            "",
            f"- 唯一主要变动：{r['single_change']}",
            f"- 证据等级：`{r['evidence_level']}`",
            f"- ⚠️ {r['warning']}",
            "",
            "| 指标 | A 基线 | B 改进 | 差值 |",
            "| --- | ---: | ---: | ---: |",
        ]
        for key, name in [("total_detections", "检出框总数"),
                          ("fps", "FPS"), ("mean_latency_ms", "平均时延 (ms)"),
                          ("failure_rate", "失败/异常率"),
                          ("confirmed_total", "确认违规累计"),
                          ("alert_events", "告警事件条数")]:
            a, b = A.get(key), B.get(key)
            d = round(b - a, 4) if isinstance(a, (int, float)) and isinstance(b, (int, float)) else "-"
            lines.append(f"| {name} | {a} | {b} | {d} |")
        if A.get("ground_truth"):
            for key, name in [("precision", "Precision"), ("recall", "Recall"),
                              ("map50", "mAP@0.5")]:
                a, b = A.get(key), B.get(key)
                d = round(b - a, 4) if isinstance(a, (int, float)) and isinstance(b, (int, float)) else "-"
                lines.append(f"| {name} | {a} | {b} | {d} |")
        else:
            lines.append("| Precision / Recall / mAP | 无真值，不输出 | 无真值，不输出 | — |")
        lines += [
            "",
            f"- **结论**：见上表数值变化；{A.get('claim_limit', '')}",
            f"- **证据**：`experiments/results/{eid}.json`（含环境指纹、冻结条件、逐项指标）",
            f"- **边界**：{r['conclusion_boundary']}",
            "",
        ]

    lines += [
        "## 结论边界总表",
        "",
        "| 不能说的 | 原因 | 下一步验证 |",
        "| --- | --- | --- |",
        "| 「准确率提升 xx%」 | 无人工逐帧真值时，检出框数量变化不等于准确率 | 标注测试集后重跑 `--backend yolo` |",
        "| 「模型达到 SOTA」 | 未做跨模型、跨数据集对照 | 引入公开基线同条件复现 |",
        "| 「现场事故率下降」 | 未接入真实工地长期运行数据 | 与项目部合作做部署后观测 |",
        "| 「实时无延迟」 | 时序投票有 K-1 帧固有延迟 | 报告端到端 P95 时延 |",
        "",
        f"> 当前已运行实验：{found}/{len(IDS)}。剩余项保持「待测」，这是证据完整性的一部分，不是缺陷。",
        "",
    ]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return out
