# 证据索引 · 让陌生人 30 秒看懂

> 页面唯一结论：**同一份真实工程资产，针对不同读者调整证据组织方式。**

## 课前诊断五问（0/1/2 自评）

| 五问 | 本项目回答 | 证据位置 | 自评 |
| --- | --- | --- | --- |
| 1 解决了什么真实问题？ | 远景小目标漏检 / 告警抖动 / 无处置闭环 | `README.md` §1、`docs/INNOVATION.md` | 2 |
| 2 相对基线改了什么？ | 六环节基线 + 圈出 3 个改动模块 | `docs/ARCHITECTURE.md`、`docs/architecture.svg` | 2 |
| 3 哪些实测数据支持？ | 三组 A/B JSON + 环境指纹 + 对比图 | `experiments/results/` | 1（仅演练数据，真实数据待测） |
| 4 本人负责哪些工作？ | 10 项贡献 + 可打开的材料 | `docs/CONTRIBUTION.md` | 1（需本人核对） |
| 5 陌生人怎样复现？ | 四条命令 + 四段式 README + 锁定依赖 | `README.md` §2 | 2 |

> 自评说明：**缺少真实真值数据**是当前最大短板，因此第 3 项只给 1 分，不粉饰。

## 三个评价舞台各自看什么

| 舞台 | 关注点 | 本项目对应材料 |
| --- | --- | --- |
| 竞赛评委 | 问题、创新、对照实验 | `docs/INNOVATION.md`、`docs/EXPERIMENT_CARD.md`、答辩 PPT |
| GitHub 陌生人 | 代码、说明、复现路径 | `README.md` 四段、`requirements.txt`、`run.py`、`configs/` |
| 企业招聘 | 个人职责、工程质量、岗位适配 | `docs/CONTRIBUTION.md`、简历 docx、90 秒自我介绍 |

## 五项成果状态（对应课程交付清单）

| 成果 | 文件 | 状态 |
| --- | --- | --- |
| 创新陈述 | `docs/INNOVATION.md` | ✅ 3 条，均含问题/行动/结果/边界 |
| 实验卡 | `docs/EXPERIMENT_CARD.md` | ✅ 由 `run.py report` 自动生成，未运行项标「待测」 |
| PPT 改稿 | 答辩 PPT（工作区根目录） | ✅ 按 7+3 与六页论证故事组织 |
| README / 仓库 | `README.md` + 本仓库 | ✅ 四段式首屏、能看能跑能对比 |
| 简历 | `刘佳豪个人简历(1).docx`（修改版） | ✅ 项目经历改为问题—行动—结果—边界 |

## 证据等级一览

| 材料 | 等级 | 说明 |
| --- | --- | --- |
| `experiments/results/E*.json`（当前） | `SELFTEST_ONLY` | demo 演练后端，禁止对外引用数值 |
| `experiments/results/E*.json`（换 yolo + 真值后） | `MEASURED` | 方可写入简历 / 答辩 |
| `data/synthetic/` | `SELFTEST_ONLY` | 合成场景，仅验证流程 |
| `docs/EXPERIMENT_CARD.md` | 跟随运行结果 | 数字由脚本生成，不手写 |
| 答辩 PPT 中的性能数字 | 待测 | 页面中已用「待测」占位，未编造 |

## 课后行动清单（课程原话）

- 系统架构图 + 执行流程图 → `docs/architecture.svg`、`docs/flowchart.svg`
- A/B 实验记录与性能对比 → `experiments/results/`
- 3 条可复核的创新点表述 → `docs/INNOVATION.md`
- 课堂要求：**能看、能跑、能对比、能复述**
