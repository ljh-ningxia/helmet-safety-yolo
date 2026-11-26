#!/usr/bin/env python3
"""Generate documentation from measured artifacts.

Documentation that contains numbers must never be hand-maintained: it drifts.
This script reads the JSON files produced by the pipeline and renders
docs/DATASET.md, docs/MODEL_CARD.md, docs/EVALUATION.md plus the auto block in
README.md. Re-run it after every training/evaluation run.

    python scripts/07_generate_docs.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from helmet_safety.logging import get_logger  # noqa: E402

log = get_logger(__name__)
R = ROOT / "experiments" / "results"
D = ROOT / "docs"


def load(name: str, default: Any = None) -> Any:
    p = R / name
    if not p.is_file():
        log.warning("artifact missing", name=name)
        return default if default is not None else {}
    return json.loads(p.read_text(encoding="utf-8"))


def md_table(headers, rows) -> str:
    out = ["| " + " | ".join(str(h) for h in headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def f(v, n=4) -> str:
    return "—" if v is None else f"{float(v):.{n}f}"


def dataset_doc() -> str:
    m_path = ROOT / "data" / "raw" / "manifest.json"
    m = json.loads(m_path.read_text(encoding="utf-8")) if m_path.is_file() else {}
    a = load("dataset_audit.json")
    p = load("dataset_preparation.json")
    splits = a.get("splits", {})
    cc = a.get("class_counts", {})
    scale = a.get("scale_buckets_px", {})
    area = a.get("bbox_area_px2", {})
    lum = a.get("image_brightness_mean", {})

    files = "\n".join(
        f"| `{r['split']}` | {r['size_bytes']/1024/1024:.1f} MB | `{r['sha256'][:16]}…` |"
        for r in m.get("files", []))
    split_rows = [
        [s, v.get("images", "—"), v.get("boxes", "—"), v.get("empty_images", "—"),
         v.get("boxes_per_image_mean", "—"), v.get("boxes_per_image_p95", "—")]
        for s, v in splits.items()]
    return f"""# Dataset Card — Hard Hats Dataset

> 本文件由 `scripts/07_generate_docs.py` 从 `experiments/results/dataset_audit.json`
> 自动生成，请勿手工修改数字。

## 1. 来源与许可

| 项 | 值 |
|---|---|
| 名称 | Hard Hats Dataset (Roboflow Universe, 2022) |
| 镜像 | https://huggingface.co/datasets/{m.get('dataset_id','keremberke/hard-hat-detection')} |
| 原始页面 | https://universe.roboflow.com/roboflow-universe-projects/hard-hats-fhbh5/dataset/2 |
| **许可** | **CC BY 4.0**（数据集卡片与每个 COCO JSON 的 `licenses` 字段双重声明） |
| 上游 revision | `{m.get('revision','main')}` / `{m.get('upstream_sha','—')}` |
| 引用义务 | 见 `NOTICE.md`；任何再分发需保留署名 |
| 类别 | `0: hardhat`（佩戴）/ `1: no-hardhat`（未佩戴，告警对象） |
| 原标注格式 | COCO JSON（每 split 一个 `_annotations.coco.json`） |
| 上游预处理 | EXIF 自动定向 + **拉伸** resize 到 640×640；未做数据增强 |

### 下载归档（SHA256 已固定）

| split | 大小 | SHA256 前缀 |
|---|---|---|
{files or '| — | — | — |'}

## 2. 实际使用规模

本项目在 CPU 环境使用固定 seed `{a.get('root','') and p.get('seed','—')}` 做了
**类别-密度分层抽样**（`src/helmet_safety/data/split.py`），保证稀有的
`no-hardhat` 样本按原比例保留。上游全量为 19,745 张图
（train 13,782 / valid 3,962 / test 2,001），实际参与本次训练/评估的是：

{md_table(['split', '图片数', '标注框', '空图', '平均每图框数', 'P95 每图框数'], split_rows)}

> 每个 split 的分层明细记录在 `experiments/results/dataset_preparation.json`
> 的 `splits[].strata` 字段，可核对抽样是否偏离原始分布。

## 3. 数据画像（实测）

{md_table(['指标', '值'], [
    ['总图片数', a.get('total_images','—')],
    ['总标注框', a.get('total_boxes','—')],
    ['hardhat 框', cc.get('hardhat','—')],
    ['no-hardhat 框', cc.get('no-hardhat','—')],
    ['类别不平衡倍数', a.get('imbalance_ratio','—')],
    ['小目标 <32² px', scale.get('small','—')],
    ['中目标 32²–96² px', scale.get('medium','—')],
    ['大目标 >96² px', scale.get('large','—')],
    ['框面积 P10 / P50 / P90 (px²)', f"{area.get('p10','—')} / {area.get('p50','—')} / {area.get('p90','—')}"],
    ['图像亮度 P10 / P50 / P90', f"{lum.get('p10','—')} / {lum.get('p50','—')} / {lum.get('p90','—')}"],
    ['暗图（均值<60）张数', lum.get('dark_lt_60','—')],
    ['亮图（均值>180）张数', lum.get('bright_gt_180','—')],
    ['完全相同图片组数（MD5）', a.get('exact_duplicate_groups','—')],
])}

图表：`experiments/results/dataset_audit_charts.png`；
真实标注样本：`experiments/results/dataset_samples.png`。

## 4. 已知偏差与边界

- 上游把不同长宽比的原图**拉伸**到 640×640，几何比例已失真；本项目的框坐标基于该失真图，迁移到自有摄像头会存在域偏移。
- 图片来自公开网络图集，不是同一工地的连续监控视频；因此**无法**评估跨摄像头泛化、时序稳定性或真实事故率。
- 亮度只是灰度均值代理，不是人工标注的「夜间/逆光」标签，切片结论仅供诊断。
- 类别不平衡（见上表）会让 `no-hardhat` 的召回天然更难，报告指标必须按类分列。
- 未做人工复核标注质量；COCO 校验只覆盖结构性问题（重复/孤儿/越界/零面积）。

## 5. 合规

- 代码 MIT；**数据 CC BY 4.0，需署名**，见 `NOTICE.md`。
- 原始图片与解压结果不进入 git（`.gitignore`），仓库只保留下载脚本、校验和与统计。
- `dataset_samples.png` 中的少量真实图片属于同一数据集，已在此文件署名。
"""


def model_card_doc() -> str:
    t = load("train_run.json")
    e = load("evaluation.json")
    o = e.get("official_validation", {})
    lat = e.get("latency_benchmark", {})
    w = t.get("config", {})
    env = t.get("environment", {})
    m_path = ROOT / "models" / "weights_manifest.json"
    wm = json.loads(m_path.read_text(encoding="utf-8")) if m_path.is_file() else {}
    per = o.get("per_class", {})
    rows = [[k, f(v.get("precision")), f(v.get("recall")),
             f(v.get("map50")), f(v.get("map50_95"))] for k, v in per.items()]
    return f"""# Model Card — YOLOv8n Hard Hat Detector

> 由 `scripts/07_generate_docs.py` 从 `train_run.json` / `evaluation.json` 生成。

## 模型概要

| 项 | 值 |
|---|---|
| 架构 | YOLOv8n (Ultralytics 8.4.x) |
| 基座权重 | `{wm.get('file','models/yolov8n.pt')}`，SHA256 `{wm.get('sha256','—')[:16]}…` |
| 基座许可 | AGPL-3.0（见 `NOTICE.md`，商用需另取授权） |
| 训练产物 | `{t.get('best_weights','—')}` |
| 类别 | `hardhat`, `no-hardhat` |
| 用途 | 工地图像/视频帧中的安全帽佩戴状态检测与未佩戴告警 |

## 训练配置（实测运行记录）

{md_table(['参数', '值'], [[k, v] for k, v in w.items()])}

- 训练耗时：`{t.get('elapsed_seconds','—')} s`，记录 epoch 数：`{t.get('epochs_logged','—')}`
- 环境：Python `{env.get('python','—')}`、torch `{env.get('torch','—')}`、
  ultralytics `{env.get('ultralytics','—')}`、设备 `{env.get('device_used','—')}`、
  CPU `{env.get('cpu_count','—')}` 核、git `{str(env.get('git_head','—'))[:8]}`
- 完整可复现命令：`{t.get('command','—')}`
- 逐 epoch 曲线：`{t.get('run_dir','—')}/results.csv`

## 独立测试集性能（MEASURED）

{md_table(['类别', 'Precision', 'Recall', 'mAP@0.5', 'mAP@0.5:0.95'], rows)
if rows else '_按类指标未生成_'}

{md_table(['总体指标', '值'], [
    ['Precision', f(o.get('precision'))],
    ['Recall', f(o.get('recall'))],
    ['mAP@0.5', f(o.get('map50'))],
    ['mAP@0.5:0.95', f(o.get('map50_95'))],
    ['测试图片数', e.get('n_test_images','—')],
    ['推理均值时延', f"{lat.get('mean_ms','—')} ms"],
    ['推理 P95 时延', f"{lat.get('p95_ms','—')} ms"],
    ['吞吐（由均值换算）', f"{lat.get('fps_from_mean','—')} FPS"],
])}

> 评估 split：`{e.get('split','test')}`，conf `{e.get('conf','—')}`，IoU `{e.get('iou','—')}`。
> 指标由 Ultralytics 官方 `model.val()` 产出，非自行估算。

## 预期用途 / 不适用场景

**适用**：静态图片或视频帧中的安全帽佩戴检测、告警策略研究、算法 A/B 对照、教学与作品展示。

**不适用**：
- 直接作为生产安全合规判定（未经目标现场数据验证）；
- 夜间/逆光/雨雾/严重遮挡的可靠性承诺（未做分层标注验证）；
- 佩戴规范性（帽带是否系好）、其他 PPE（反光衣/护目镜）识别；
- 边缘设备实时性承诺（时延为 CPU 单图同步推理，无 TensorRT/ONNX 优化）。

## 伦理与风险

- 系统输出是**辅助预警**，不是对人的处分依据；误报会干扰现场，漏报有安全风险。
- 涉及人员影像，部署需遵守当地隐私与 workplace monitoring 规定，做最小化留存与访问控制。
- 类别不平衡会使少数类召回偏低，上线前必须用现场数据重新校准阈值。
"""


def evaluation_doc() -> str:
    e = load("evaluation.json")
    ab = load("ab_summary.json")
    o = e.get("official_validation", {})
    lat = e.get("latency_benchmark", {})
    slices = e.get("slice_metrics", {})
    slice_rows = [[k, v.get("images", "—"), f(v.get("precision")), f(v.get("recall")),
                   f(v.get("f1")), f"{v.get('tp',0)}/{v.get('fp',0)}/{v.get('fn',0)}"]
                  for k, v in sorted(slices.items())]
    exp_rows = []
    for r in ab.get("experiments", []):
        A, B = r.get("A", {}), r.get("B", {})
        exp_rows.append([
            f"`{r.get('experiment','—')}`",
            f"R {f(A.get('recall'))} / {f(A.get('mean_latency_ms'),1)} ms / {A.get('alert_events','—')} 事件",
            f"R {f(B.get('recall'))} / {f(B.get('mean_latency_ms'),1)} ms / {B.get('alert_events','—')} 事件",
            r.get("conclusion_boundary", "—"),
        ])
    fails = e.get("failure_cases_top10", [])
    fail_rows = [[x.get("image", "—"), x.get("gt", "—"), x.get("pred", "—"),
                  x.get("fp", "—"), x.get("fn", "—"), x.get("luminance", "—")]
                 for x in fails[:8]]
    boundaries = "\n".join(f"- {b}" for b in e.get("boundaries", []))
    return f"""# 评估报告（全部为实测）

> 由 `scripts/07_generate_docs.py` 生成。原始数据：
> `experiments/results/evaluation.json`、`benchmark.json`、`ab_summary.json`。

## 1. 官方验证（held-out `{e.get('split','test')}` split）

{md_table(['指标', '实测值'], [
    ['Precision', f(o.get('precision'))],
    ['Recall', f(o.get('recall'))],
    ['mAP@0.5', f(o.get('map50'))],
    ['mAP@0.5:0.95', f(o.get('map50_95'))],
    ['验证耗时', f"{o.get('val_seconds','—')} s"],
    ['测试图片数', e.get('n_test_images','—')],
])}

## 2. 延迟基准

{md_table(['指标', '实测值'], [
    ['设备', lat.get('device','—')],
    ['输入尺寸', lat.get('imgsz','—')],
    ['预热 / 重复次数', f"{lat.get('warmup','—')} / {lat.get('repeats','—')}"],
    ['均值', f"{lat.get('mean_ms','—')} ms"],
    ['P50 / P95', f"{lat.get('p50_ms','—')} / {lat.get('p95_ms','—')} ms"],
    ['换算 FPS', lat.get('fps_from_mean','—')],
])}

> {lat.get('note','')}

## 3. 场景切片（诊断用，非 mAP）

方法：{e.get('slice_method','—')}

{md_table(['切片', '图片数', 'Precision', 'Recall', 'F1', 'TP/FP/FN'], slice_rows)}

切片用于发现「哪类场景更差」，不等价于 COCO mAP，也不用于对外宣称精度。

## 4. 单变量公平 A/B

冻结条件：同一权重、同一批真实图片、同一顺序、同一 seed；每次仅改一个开关。
完整 JSON：`experiments/results/ab_*.json`。

{md_table(['实验', 'A 基线', 'B 变体', '结论边界'], exp_rows)}

## 5. 失败案例（误差最大的真实图片）

{md_table(['图片', 'GT 框', '预测框', 'FP', 'FN', '亮度'], fail_rows)}

可视化证据：`assets/real_inference.png`（绿框真值 / 红框预测）。

## 6. 结论边界

{boundaries or '- 见 docs/BOUNDARIES.md'}
"""


START = "<!-- AUTO:RESULTS START -->"
END = "<!-- AUTO:RESULTS END -->"


def patch_readme() -> None:
    p = ROOT / "README.md"
    if not p.is_file():
        log.warning("README missing, skip patch")
        return
    s = p.read_text(encoding="utf-8")
    e = load("evaluation.json")
    t = load("train_run.json")
    a = load("dataset_audit.json")
    o = e.get("official_validation", {})
    lat = e.get("latency_benchmark", {})
    block = f"""{START}
### 实测结果（自动生成，勿手改）

| 指标 | 实测值 |
|---|---|
| 训练数据 | {a.get('total_images','—')} 张真实图（train/valid/test 分层子集） |
| 训练 epoch | {t.get('epochs_logged','—')} · CPU `{t.get('elapsed_seconds','—')} s` |
| mAP@0.5 | **{f(o.get('map50'))}** |
| mAP@0.5:0.95 | **{f(o.get('map50_95'))}** |
| Precision / Recall | {f(o.get('precision'))} / {f(o.get('recall'))} |
| 独立测试图片 | {e.get('n_test_images','—')} |
| CPU 推理 P95 | {lat.get('p95_ms','—')} ms |
| CPU 吞吐 | {lat.get('fps_from_mean','—')} FPS |

来源：`experiments/results/{{train_run,evaluation,benchmark,dataset_audit}}.json`。
运行 `python scripts/07_generate_docs.py` 可重新生成。
{END}"""
    if START in s and END in s:
        s = s[:s.index(START)] + block + s[s.index(END) + len(END):]
    else:
        s = s.rstrip() + "\n\n" + block + "\n"
    p.write_text(s, encoding="utf-8")
    log.info("README auto block updated")


def main() -> int:
    D.mkdir(parents=True, exist_ok=True)
    (D / "DATASET.md").write_text(dataset_doc(), encoding="utf-8")
    (D / "MODEL_CARD.md").write_text(model_card_doc(), encoding="utf-8")
    (D / "EVALUATION.md").write_text(evaluation_doc(), encoding="utf-8")
    patch_readme()
    print("generated: docs/DATASET.md, docs/MODEL_CARD.md, docs/EVALUATION.md, README metrics block")
    return 0


if __name__ == "__main__":
    sys.exit(main())
