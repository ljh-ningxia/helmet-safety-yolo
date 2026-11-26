# 系统架构说明（v2 · 真实数据版）

> 本文件描述**当前代码**的实际结构，不是设计愿景。所有模块路径均可在
> `src/helmet_safety/` 与 `scripts/` 中逐一核对。
> 配套阅读：[DATASET.md](DATASET.md) · [MODEL_CARD.md](MODEL_CARD.md) ·
> [EVALUATION.md](EVALUATION.md) · [BOUNDARIES.md](BOUNDARIES.md) ·
> [REPRODUCE.md](REPRODUCE.md) · [ADR/](ADR/)

---

## 1. 架构目标

v1 的核心问题是「数字来自演练后端」。v2 的架构目标因此只有一个：

> **任何对外出现的数字，都必须能沿着一条明确的链路回溯到一次真实运行。**

由此推导出四条硬约束：

| 约束 | 落地方式 |
|---|---|
| 无假数据兜底 | `YoloDetector` 是唯一推理后端；权重缺失直接抛错，API 返回 503 |
| 数字不手写 | `scripts/07_generate_docs.py` 从 `experiments/results/*.json` 渲染 README / 数据卡 / 模型卡 / 评估报告 / PPT |
| 改进可归因 | 三个推理开关相互独立，A/B 每次只改一个变量 |
| 条件可复现 | 固定 seed、固定 revision、SHA256 清单、环境指纹、逐 epoch 日志 |

---

## 2. 分层结构

```text
┌──────────────────────────────────────────────────────────────────┐
│ 交付层  README / docs / 答辩 PPT / Web 看板                        │
│         全部由 experiments/results/*.json 驱动，不维护手写指标       │
├──────────────────────────────────────────────────────────────────┤
│ 服务层  api/app.py (FastAPI)     cli.py (doctor/info/detect/serve) │
│         web/index.html           Dockerfile / compose / Makefile   │
├──────────────────────────────────────────────────────────────────┤
│ 业务层  inference/pipeline.py → tiling / temporal / alert/policy   │
│         schemas.py (FrameResult / Detection / AlertEvent / BBox)   │
├──────────────────────────────────────────────────────────────────┤
│ 推理层  inference/detector.py (Ultralytics YOLOv8n 唯一后端)        │
├──────────────────────────────────────────────────────────────────┤
│ 数据层  data/coco · split · convert · audit   eval/metrics         │
│         config.py (typed dataclass)   logging.py (结构化日志)       │
├──────────────────────────────────────────────────────────────────┤
│ 流水线  scripts/00 → 07   权重/数据/训练/评估/A-B/文档               │
└──────────────────────────────────────────────────────────────────┘
```

依赖方向严格自上而下；下层不读取上层配置，业务代码不解析裸环境变量。

### 模块职责

| 模块 | 职责 | 关键接口 |
|---|---|---|
| `config.py` | 集中加载并校验 5 份 YAML，产出不可变 `AppConfig` | `load_config(path, overrides, env)` |
| `logging.py` | 结构化日志；`HELMET_LOG_JSON=1` 输出 JSON 行 | `get_logger(__name__)` · `ContextAdapter` |
| `schemas.py` | 领域对象与证据等级，禁止裸 dict 跨层传递 | `BBox` / `Detection` / `FrameResult` / `EvidenceLevel` |
| `data/coco.py` | 解析 COCO JSON 并统计结构性问题 | `load_coco()` |
| `data/split.py` | 类别-密度分层抽样（最大余数法，固定 seed） | `stratified_indices()` |
| `data/convert.py` | COCO xywh → YOLO cxcywh，归一化并裁剪 | `convert_split()` |
| `data/audit.py` | 真实像素画像、重复图检测、证据图 | `audit_yolo()` |
| `eval/metrics.py` | 类内贪心 IoU@0.5 匹配与 P/R/F1 聚合 | `greedy_match()` · `aggregate()` |
| `inference/detector.py` | 唯一真实推理后端，权重懒加载 | `YoloDetector.predict()` |
| `inference/tiling.py` | 重叠切片、坐标还原、类内 NMS | `make_tiles()` · `tiled_predict()` |
| `inference/temporal.py` | IoU 轨迹关联 + 窗口多数投票 | `TemporalVoter.update()` |
| `alert/policy.py` | L1/L2/L3 分级、冷却去重、可插拔 sink | `AlertManager.emit()` |
| `inference/pipeline.py` | 串起端到端链路，三个开关独立 | `SafetyPipeline.process()` |
| `api/app.py` | REST 接口与看板托管 | 见 §5 |

---

## 3. 数据契约

### 3.1 输入

上游 `keremberke/hard-hat-detection`（CC BY 4.0）提供 4 个 zip，每 split 一份
COCO `_annotations.coco.json`。`scripts/01_download_dataset.py` 用 HTTP Range
断点续传下载并写 `data/raw/manifest.json`，记录 `revision`、`upstream_sha`、
逐文件 SHA256 与许可信息。

### 3.2 中间的 YOLO 视图

```text
data/processed/
├── images/{train,valid,test}/*.jpg
├── labels/{train,valid,test}/*.txt     # cls cx cy w h，归一化到 [0,1]
└── data.yaml                           # path / train / val / test / names
```

转换前执行四类结构校验：重复 image/annotation id、孤儿标注、越界框、
零面积框/缺失图片。重跑时旧产物**归档**到 `data/interim/stale_*` 而非删除，
保证任何一次实验的子集都能被追溯。

### 3.3 输出（唯一事实来源）

| 文件 | 生产者 | 消费者 |
|---|---|---|
| `experiments/results/train_run.json` | `03_train.py` | 模型卡 / README / PPT |
| `experiments/results/evaluation.json` | `04_evaluate.py` | 评估报告 / 看板 / PPT |
| `experiments/results/benchmark.json` | `04_evaluate.py` | 评估报告 / 看板 |
| `experiments/results/ab_*.json`、`ab_summary.json` | `05_run_ab.py` | 看板 / PPT / 评估报告 |
| `experiments/results/dataset_audit.json` | `02_prepare_dataset.py` | 数据卡 / 看板 / PPT |
| `assets/real_inference.png` | `04_evaluate.py` | 看板 / PPT |

所有指标 JSON 都带 `evidence_level: MEASURED`。v2 已从代码中移除
`SELFTEST_ONLY` 结果路径——**系统里不存在可以产生「演练指标」的分支**。

---

## 4. 推理链路

```text
frame (BGR ndarray)
   │
   ├─ enable_tiling ? tiled_predict(detector, frame, 320, 0.25, iou)
   │                    └─ 切块 → 逐块 YOLO → 坐标偏移还原 → 类内 NMS
   │                  : detector.predict(frame)            [整图单次 YOLO]
   │
   ├─ enable_temporal ? TemporalVoter.update(dets)
   │                    └─ IoU 关联轨迹 → 窗口内 ≥3 票才输出
   │                  : list(dets)                        [原始检出]
   │
   └─ AlertManager.emit(confirmed)
        └─ cls==1 (no-hardhat) 计数 → L1/L2/L3 → 冷却去重 → sink (JSONL / 日志)
   │
   └─ FrameResult { frame_id, detections, confirmed, alerts,
                    latency_ms, source, evidence_level=MEASURED }
```

三个开关分别由 `configs/inference/default.yaml` 的
`enable_tiling` / `enable_temporal` 与 `PolicyConfig.enable_dedup` 控制，
彼此无耦合——这是单变量 A/B 能成立的结构前提。

**失败语义**：权重文件不存在时 `YoloDetector.model` 抛 `DetectorError`，
`POST /api/v1/detect` 返回 HTTP 503 并说明原因；不会静默降级到任何演示检测器。

---

## 5. 服务接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 存活与权重可用性 |
| GET | `/api/v1/config` | 当前生效配置（`SafetyPipeline.describe()`） |
| GET | `/api/v1/metrics` | `evaluation.json` + `benchmark.json` |
| GET | `/api/v1/dataset` | `dataset_audit.json` + `manifest.json` |
| GET | `/api/v1/ab` | `ab_summary.json` |
| GET | `/api/v1/samples` | 真实推理证据图索引 |
| POST | `/api/v1/detect` | 上传图片 → 框 / 告警 / 标注图；权重缺失返回 503 |
| GET | `/` | Web 看板（数字全部由上述接口读取） |

CLI：`python -m helmet_safety.cli doctor | info | detect <image> --out | serve`。

---

## 6. 可复核性设计

1. **固定条件**：`seed=20261126`、上游 revision 与 SHA256、基座权重 SHA256、
   `imgsz/batch/epochs/conf/iou/device` 全部写入 `train_run.json` 的 `config` 与
   `environment` 指纹（Python / torch / ultralytics / CPU 核数 / git HEAD）。
2. **冻结 A/B**：同一权重、同一批图片、同一顺序、同一 seed，每组只改一个开关；
   每个 `ab_*.json` 自带 `frozen_conditions` 与 `conclusion_boundary`。
3. **数字单一来源**：`07_generate_docs.py` 渲染所有对外文档；PPT 构建脚本
   `_build_slides.py` 也直接读取同一批 JSON。
4. **原始样本保留**：`benchmark.json` 存逐次延迟样本，`evaluation.json` 存
   `failure_cases_top10`，训练 `results.csv` 存逐 epoch 曲线。
5. **合规隔离**：原始图片、解压结果、`*.pt` 权重不进入 git；代码 MIT、
   数据 CC BY 4.0、基座权重 AGPL-3.0 在 `NOTICE.md` 分别声明。

---

## 7. 部署形态

- 本地：`make install && make data && make train && make eval && make ab`，
  或 `make serve` 起 FastAPI。
- 容器：`Dockerfile`（`python:3.13-slim` + `libgl1/libglib2.0-0/libgomp1`）
  与 `docker-compose.yml`；`models/`、`data/`、`experiments/`、`assets/`
  通过 volume 挂载，镜像不烘焙大文件。
- CI：`.github/workflows/ci.yml` 两个 job —— `test`（ruff / pytest / CLI / 脚本编译）
  与 `docs`（强制检查 9 份必需文档非空）；**CI 不依赖权重与数据**，因此可在
  干净环境快速反馈。

---

## 8. 已知取舍

| 取舍 | 理由 | 代价 |
|---|---|---|
| CPU 上用分层子集（2,000/600/600）而非全量 19,745 张 | 由 1-epoch 速度探测实测确定，保证「真的跑得完」 | 精度不是该数据集上限；见 ADR-0001 |
| 无 demo fallback | 避免演练数字混入报告 | 未下载权重时服务不可用，需要显式跑流水线 |
| 时序投票用离散图片序列做 A/B | 数据集无连续视频 | 只能证明机制对输出数量的影响，不能证明视频稳定性 |
| 切片尺寸固定 320 / overlap 0.25 | 与训练 `imgsz=320` 对齐，便于单变量对照 | 未做切片尺度寻优 |
| 结构化日志而非指标系统 | 单机作品项目，避免过度工程 | 尚无 Prometheus / 分布式追踪 |

架构决策的完整背景见 [ADR/0001–0004](ADR/)。所有「暂不能证明什么」写在
[BOUNDARIES.md](BOUNDARIES.md)，不在此处用架构描述掩盖。
