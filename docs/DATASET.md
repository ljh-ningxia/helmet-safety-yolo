# Dataset Card — Hard Hats Dataset

> 本文件由 `scripts/07_generate_docs.py` 从 `experiments/results/dataset_audit.json`
> 自动生成，请勿手工修改数字。

## 1. 来源与许可

| 项 | 值 |
|---|---|
| 名称 | Hard Hats Dataset (Roboflow Universe, 2022) |
| 镜像 | https://huggingface.co/datasets/keremberke/hard-hat-detection |
| 原始页面 | https://universe.roboflow.com/roboflow-universe-projects/hard-hats-fhbh5/dataset/2 |
| **许可** | **CC BY 4.0**（数据集卡片与每个 COCO JSON 的 `licenses` 字段双重声明） |
| 上游 revision | `main` / `694c61350faf9a6622586d6cf50f45e1631862dc` |
| 引用义务 | 见 `NOTICE.md`；任何再分发需保留署名 |
| 类别 | `0: hardhat`（佩戴）/ `1: no-hardhat`（未佩戴，告警对象） |
| 原标注格式 | COCO JSON（每 split 一个 `_annotations.coco.json`） |
| 上游预处理 | EXIF 自动定向 + **拉伸** resize 到 640×640；未做数据增强 |

### 下载归档（SHA256 已固定）

| split | 大小 | SHA256 前缀 |
|---|---|---|
| `test` | 109.1 MB | `e21ca22f4d69411a…` |
| `valid-mini` | 0.2 MB | `cc44c6d3c70c51cf…` |
| `train` | 742.8 MB | `091b305ae044b652…` |
| `valid` | 214.4 MB | `3d39f51b1e9cd968…` |

## 2. 实际使用规模

本项目在 CPU 环境使用固定 seed `20261126` 做了
**类别-密度分层抽样**（`src/helmet_safety/data/split.py`），保证稀有的
`no-hardhat` 样本按原比例保留。上游全量为 19,745 张图
（train 13,782 / valid 3,962 / test 2,001），实际参与本次训练/评估的是：

| split | 图片数 | 标注框 | 空图 | 平均每图框数 | P95 每图框数 |
|---|---|---|---|---|---|
| train | 2000 | 5617 | 32 | 2.808 | 9.0 |
| valid | 600 | 1662 | 8 | 2.77 | 9.0 |
| test | 600 | 1649 | 8 | 2.748 | 8.05 |

> 每个 split 的分层明细记录在 `experiments/results/dataset_preparation.json`
> 的 `splits[].strata` 字段，可核对抽样是否偏离原始分布。

## 3. 数据画像（实测）

| 指标 | 值 |
|---|---|
| 总图片数 | 3200 |
| 总标注框 | 8928 |
| hardhat 框 | 6913 |
| no-hardhat 框 | 2015 |
| 类别不平衡倍数 | 3.43 |
| 小目标 <32² px | 1419 |
| 中目标 32²–96² px | 5648 |
| 大目标 >96² px | 1861 |
| 框面积 P10 / P50 / P90 (px²) | 756.3 / 3157.0 / 20269.9 |
| 图像亮度 P10 / P50 / P90 | 87.9 / 123.1 / 166.3 |
| 暗图（均值<60）张数 | 86 |
| 亮图（均值>180）张数 | 161 |
| 完全相同图片组数（MD5） | 0 |

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
