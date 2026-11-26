# 个人贡献清单 · 可核验材料

> 页面唯一结论：**每条贡献都指向一份能被打开、运行、复核的材料。**
> ⚠️ 本清单由项目结构整理而成，用于答辩与简历自检。请按本人**实际参与情况**逐条核对、删改，
> 不要照抄不属于自己的工作 —— 课程规范明确要求「AI 协助，人来验收」。

## 目标岗位与能力对齐

**目标岗位**：视觉算法工程师 / AI 工程师（实习）

| 岗位能力 | 本人项目证据 |
| --- | --- |
| 误检漏检分析 | `docs/INNOVATION.md` ①、`experiments/results/E1_tiling.json` 逐类 TP/FP/FN |
| 评估脚本与真实错误样例 | `src/metrics.py`、`experiments/results/E1_tiling.png`、`docs/EXPERIMENT_CARD.md` |
| 数据 / 训练 / 评测 / 部署 | `configs/data.yaml`、`scripts/train.py`、`run.py ab`、`run.py web` |
| 需求 / 交互 / 可靠性与文档 | `src/webapp/`、`src/alert.py`、`README.md` 四段式、`docs/` 全套 |

## 逐项贡献与证据路径

| # | 本人负责的工作 | 打开哪份材料复核 | 复核方式 |
| --- | --- | --- | --- |
| 1 | 梳理工地安全监管痛点，定义「远景小目标 / 告警抖动 / 无处置闭环」三个真实问题 | `README.md` §1、`docs/INNOVATION.md` | 逐条对应到可观察现象，不是泛泛而谈 |
| 2 | 画出系统基线六环节，并圈出本人改动的三个模块 | `docs/ARCHITECTURE.md`、`docs/architecture.svg` | 基线在前、改动在后 |
| 3 | 实现重叠切片推理（含坐标回映与跨切片 NMS） | `src/tiling.py` | `python run.py ab` 看 E1 |
| 4 | 实现 IoU 跨帧关联 + 连续 K 帧投票判定 | `src/temporal.py` | E2；注意收益不明显已被如实记录 |
| 5 | 实现分级预警、冷却去重、证据快照、JSONL 台账 | `src/alert.py` | E3 + `experiments/results/alerts.jsonl` |
| 6 | 设计并实现公平 A/B 框架（单变量、冻结条件、环境指纹） | `experiments/run_ab.py`、`configs/ab_experiment.yaml` | 每组 A/B 由同一配置派生 |
| 7 | 实现评估模块与「无真值拒绝输出 P/R/mAP」保护 | `src/metrics.py`、`tests/test_core.py` | `python run.py selftest` |
| 8 | 搭建实时预警看板与证据等级水印 | `src/webapp/`、`run.py web` | 浏览器打开 127.0.0.1:5000 |
| 9 | 撰写四段式 README、实验卡、结论边界与复现清单 | `README.md`、`docs/EXPERIMENT_CARD.md`、`docs/BOUNDARIES.md` | 30 秒体检：陌生人能否复现 |
| 10 | 对齐岗位改写简历项目经历、准备 7+3 答辩 | 简历 docx、答辩 PPT | 问题—行动—结果—边界 四段式 |

## 明确不属于本人的部分（避免夸大）

- 公开数据集的采集与标注（本项目仅提供接入规范，未分发、未标注实拍数据）；
- YOLOv8 模型结构与预训练权重（来自 ultralytics 官方，遵守其许可）；
- 现场真实部署与长期运行数据（尚未进行，已列入下一步）。

## 提交留痕

Git 提交建议按「基线 → 创新① → 创新② → 创新③ → 实验 → 文档」分步提交，
每步一条可查证的 commit；答辩被问「具体改动在哪段代码」时直接指向文件与提交。
