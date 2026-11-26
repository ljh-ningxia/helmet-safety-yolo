# 从零复现指南

所有步骤都在本仓库根目录执行。预期产物写在每一步后面，便于逐步核对。

## 0. 前置

- Python 3.11+（本项目的实测环境为 3.13.14 / Windows 10 / 12 核 CPU / 无 GPU）
- 约 3 GB 可用磁盘（原始归档约 1.07 GB，解压后约 2 GB）
- 能访问 `huggingface.co` 与 `github.com`；若网络存在 TLS 代理，
  `00_fetch_weights.py` 已改用系统通道绕开 `requests`/certifi 的证书问题

```bash
python -m pip install -r requirements-dev.txt
python -m helmet_safety.cli doctor || true   # 此时缺产物是正常的
```

## 1. 获取权重与数据

```bash
python scripts/00_fetch_weights.py
python scripts/01_download_dataset.py
python scripts/01_download_dataset.py --verify-only
```

预期产物：

| 文件 | 说明 |
|---|---|
| `models/yolov8n.pt` | 6.5 MB，YOLOv8n 基座权重 |
| `models/weights_manifest.json` | SHA256 + 来源 + 许可 |
| `data/raw/{train,valid,valid-mini,test}.zip` | 原始归档 |
| `data/raw/manifest.json` | 每个归档的 SHA256、上游 revision、许可与引用 |

下载器支持 HTTP Range 断点续传；中断后重跑同一条命令即可继续。
实测中 `train.zip` 曾断流两次，均由续传自动恢复。

## 2. 数据工程

```bash
python scripts/02_prepare_dataset.py
```

依次执行：校验和 → 解压 → COCO 结构校验 → 类别-密度分层抽样 →
COCO→YOLO 转换 → `data.yaml` → EDA 审计。

预期产物：

| 文件 | 说明 |
|---|---|
| `data/processed/{images,labels}/{train,valid,test}/` | YOLO 训练输入 |
| `data/processed/data.yaml` | Ultralytics 数据集描述 |
| `experiments/results/dataset_preparation.json` | 每个 split 的分层明细与转换计数 |
| `experiments/results/dataset_audit.json` | 实测数据画像 |
| `experiments/results/dataset_audit_charts.png` | 分布图 |
| `experiments/results/dataset_samples.png` | 真实图片 + GT 框拼图 |

抽样 seed 固定在 `configs/data/default.yaml`。把 `subset_*` 改为 `null`
即可使用全量语料（GPU 环境建议）。

> 重复运行是安全的：上一轮的产物会被归档到 `data/interim/stale_*`，
> 不会与新子集混合，也不会执行批量删除。

## 3. 训练

```bash
python scripts/03_train.py
# 可选：先在 1 epoch 上测本机吞吐
python scripts/03_train.py --smoke --data data/smoke/data.yaml --epochs 1 --workers 0
```

预期产物：`experiments/runs/yolov8n-hardhat/`（含 `weights/best.pt`、
`results.csv`、训练曲线）与 `experiments/results/train_run.json`。

后者记录：完整命令、全部超参、Python/torch/ultralytics/OpenCV 版本、
CPU 核数、CUDA 可用性、git commit、逐 epoch 最终行、Ultralytics 指标。

**CPU 预算说明**：实测速度探测为 1.25 it/s（batch 8 / imgsz 320），
因此正式配置取 2,000 张 × 12 epochs。这是「能真实跑完」的规模，
不是精度上限；换 GPU 后应把子集设为 null 重跑并对照。

## 4. 评估

```bash
python scripts/04_evaluate.py
```

预期产物：

| 文件 | 说明 |
|---|---|
| `experiments/results/evaluation.json` | 官方 mAP + 场景切片 + 失败案例 |
| `experiments/results/benchmark.json` | 每一次推理的原始毫秒数 |
| `assets/real_inference.png` | 真实图片上 GT（绿）vs 预测（红） |

评估只用 `test` split，与训练/验证完全隔离。`docs/EVALUATION.md` 会把
`best.pt` 选参所用的 valid 指标与 test 指标并列，显式给出泛化落差；
对外只引用 test 数字。

## 5. 公平 A/B

```bash
python scripts/05_run_ab.py --limit 100
```

预期产物：`experiments/results/ab_E*.json`（4 组）、`ab_summary.json`、
`ab_comparison.png`。每组 A/B 只有 `variable` 指明的那一个开关不同，
冻结条件（权重 / 图片列表 / 顺序 / seed / imgsz / IoU / device）写在每份 JSON 里。

**E3 时序投票是受控扰动实验**，与其余三组不同，理由写在这里避免复现时误读：
真实连续视频不在本数据集中，若把 100 张互不相关的静态图片直接送进
IoU 轨迹投票器，轨迹永远无法关联，recall 会无意义地变成 0。因此 E3
对每张真实图片只推理一次，再按固定 seed 构造 5 帧片段（每帧丢弃 30%
**真实检出框**，模拟瞬时漏检）；A 取该片段最后一帧，B 做多数投票。
两臂看到完全相同的帧序列，唯一差异是投票开关。注入比例、帧数、seed
都记录在 `frozen_conditions.temporal_probe` 中，检出框本身不是合成的。

> 实测结论包含负结果：E1 切片在本数据分布上召回下降且时延约 9 倍，
> 因此 `configs/inference/default.yaml` 默认关闭切片。报告不隐藏这类结果。

## 6. 生成文档

```bash
python scripts/07_generate_docs.py
```

会重写 `docs/DATASET.md` / `docs/MODEL_CARD.md` / `docs/EVALUATION.md`
以及 README 的 `AUTO:RESULTS` 区块；同时把结果 JSON 里的本机绝对路径
规范化为项目相对路径，保证公开仓库里不出现机器专属路径。
**任何数字都不要手工编辑这些文件。**

## 7. 运行服务

```bash
python -m helmet_safety.cli serve --port 8000
# 或
docker build -t helmet-safety .
docker run -p 8000:8000 -v "$PWD/models:/app/models" -v "$PWD/experiments:/app/experiments" helmet-safety
```

接口：`GET /health`、`/api/v1/config`、`/api/v1/metrics`、
`/api/v1/dataset`、`/api/v1/ab`、`POST /api/v1/detect`（ multipart 图片）。

权重缺失时 `/detect` 返回 503 并给出训练命令，**不会**静默使用假检测器。

## 8. 一致性校验

```bash
pytest -v
ruff check src scripts tests
python -m helmet_safety.cli doctor
python scripts/01_download_dataset.py --verify-only
git status --short          # 应无意外的数据/权重文件被跟踪
```
