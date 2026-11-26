# 基于 YOLO 的工地现场安全帽识别与预警系统

> **一句话价值**：把「检测到了」变成「可判定、可预警、可追溯、可复核」的工地安全闭环。
> **适用范围**：高位固定摄像头 / 施工现场出入口 / 塔吊俯拍画面的安全帽佩戴合规巡检。
> **不适用范围**：夜间低照度、强逆光、遮挡严重、快速移动目标；跨场景泛化未经真实数据验证。

---

## ⚠️ 证据等级声明（请先读这一段）

| 后端 | 说明 | 证据等级 | 能否对外引用 |
| --- | --- | --- | --- |
| `yolo` | 加载**本人训练**的 `best.pt`，在带人工真值的实拍测试集上推理 | `MEASURED` | ✅ 可以 |
| `demo` | HSV 颜色分割演练器，用于验证「链路能不能跑通、看板能不能用」 | `SELFTEST_ONLY` | ❌ **不可以** |

本仓库**不内置任何模型权重**，也不会替你编造 mAP / FPS。默认 `demo` 后端跑出来的所有数字（包括 `experiments/results/*.json`）只证明系统可运行、A/B 框架可复核，**不是 YOLO 模型性能，禁止写进简历、答辩结论或竞赛材料**。

想拿到可引用的数据，只有一条路：准备带真值的实拍测试集 + 自己训练的权重，然后执行

```bash
python run.py ab --backend yolo --weights runs/train/exp/weights/best.pt \
                 --source data/real/images --labels-dir data/real/labels --frames 300
python run.py report      # 由运行结果自动生成实验卡
```

---

## 1. 项目简介 · 解决什么问题

工地安全帽监管长期依赖人工巡查，高位摄像头下安全帽往往只有 20×20 像素，整图缩放推理后极易漏检；单帧误检又会造成告警风暴，现场人员迅速脱敏；更关键的是，多数检测项目止步于「画个框」，检测结果没有接到处置动作，事后无法追溯。

本项目先画出完整系统基线，再**只改动三个环节**，并且每个改动各自做一次单变量 A/B 对照：

| # | 真实痛点 | 改动环节 | 具体改动 | 代价 |
| --- | --- | --- | --- | --- |
| ① | 远景小目标被尺度压缩后漏检 | 定位 / 检测 | 整图一次推理 → 重叠切片 + 放大 + 跨切片 NMS | 推理次数增加，时延上升 |
| ② | 单帧误检导致告警忽明忽暗 | 判定 | 单帧即报 → 连续 K 帧命中确认 | 引入 K-1 帧固有延迟 |
| ③ | 检测完没有处置闭环、事件不可回溯 | 业务动作 | 分级预警 + 冷却去重 + 证据快照 + JSONL 台账 + 可选 Webhook | 磁盘 / 网络开销 |

**强主张**：针对高位俯拍远景小目标漏检问题，在定位/检测模块把整图推理改为重叠切片推理，在同一 60 帧合成测试集、同一阈值、同一硬件下观察到 Recall 0.45 → 0.825，但 FPS 30.99 → 3.37、mAP@0.5 0.6613 → 0.5632；目前**缺少真实工地逐帧真值**，因此不声称准确率提升，也不声称实时性达标。

---

## 2. 快速开始 · 环境与运行

**环境**：Python 3.10+（开发验证于 3.13.14 / Windows 10 / OpenCV 5.1.0），CPU 即可跑通演练链路；正式训练建议 NVIDIA GPU + CUDA。

```bash
git clone https://github.com/ljh-ningxia/helmet-safety-yolo.git
cd helmet-safety-yolo
python -m pip install -r requirements.txt      # 核心依赖；YOLO 依赖见下
python run.py selftest                        # ① 端到端链路自检
python run.py ab                              # ② 三组公平 A/B 实验
python run.py report                          # ③ 生成实验卡 docs/EXPERIMENT_CARD.md
python run.py web                             # ④ 实时预警看板 http://127.0.0.1:5000
```

正式后端额外安装：`python -m pip install ultralytics`（会带入 PyTorch）。

**输入 / 输出**

| 入口 | 输入 | 输出 |
| --- | --- | --- |
| `run.py selftest` | 固定 seed 合成帧 | 控制台自检结论；不产生性能数据 |
| `run.py ab` | 合成帧 / `data/real/images` + 标签 | `experiments/results/E*.json`、对比图、`summary.json` |
| `run.py detect --source ...` | 图片 / 视频 / 摄像头 | 带框证据图 `experiments/results/evidence.jpg` |
| `run.py web` | 合成流 / 摄像头 / RTSP | 浏览器实时画面、告警列表、指标卡、A/B 表 |
| `run.py synthetic` | — | `data/synthetic/`（图片 + YOLO 标签 + manifest） |

**看板演示**：页面三个开关分别对应三项改动，**一次只打开一个**就是在复现单变量 A/B；顶部常驻证据等级水印，避免演练截图被误当成实验结果。

---

## 3. 结果与测试 · 真实结果与边界

### 3.1 公平 A/B 的冻结条件

| 设置项 | 冻结值 |
| --- | --- |
| 数据 | 同一帧序列（固定 seed 20261126），固定划分 |
| 配置 | 同一置信度阈值 0.25、NMS IoU 0.45、同一代码基线 |
| 设备 | 同一硬件、Python / OpenCV / NumPy 版本（环境指纹写入 JSON） |
| 指标 | 有真值时 P/R/mAP@0.5；同时测 FPS、平均时延、失败/异常率、事件条数 |
| 版本 | A、B 由同一个 `PipelineConfig` 派生，**仅引入一项主要改动** |

### 3.2 本次运行结果（`demo` 后端 · `SELFTEST_ONLY` · 60 帧合成数据）

| 实验 | A 基线 | B 改进 | 怎么读 |
| --- | --- | --- | --- |
| E1 切片推理 | 检出 360，Recall 0.450，FPS 30.99 | 检出 708，Recall 0.825，FPS 3.37 | 召回上来了，但**误检也上来了**（mAP 0.6613 → 0.5632），且慢了约 9 倍 |
| E2 时序投票 | 事件 120 条 | 事件 118 条 | 稳定合成场景中收益**几乎为零**，只体现 2 帧确认延迟；真实误检分布下是否改善，待测 |
| E3 冷却去重 | 事件 120 条 | 事件 1 条 | 去重压低的是**重复上报**，不是违规人数；这不等于现场更安全 |

> 完整数据、环境指纹、逐类 TP/FP/FN 见 `experiments/results/E1_tiling.json` 等；实验卡由 `run.py report` 自动生成。
> **准确性类指标**（P/R/mAP）由固定 seed 的合成数据决定，可复现；**FPS / 时延**受机器负载影响会有波动，引用时一律以当次 `E*.json` 为准，不要照抄文档里的旧值。

### 3.3 失败案例与结论边界

- **检出框变多 ≠ 准确率提升**：E1 中 B 的检出框从 360 涨到 708，Precision 却从 0.500 降到 0.476，mAP 也下降；只报「检测数量 +97%」是错误结论。
- **无真值时系统拒绝输出 P/R/mAP**：`src/metrics.py` 在 `gts=None` 时把三项指标置为 `None` 并写入 `insufficient_ground_truth=True`。
- **合成数据不是工地数据**：`data/synthetic/` 仅用于跑通流程，任何由此得出的精度都不可对外引用。
- **未测项**：夜间 / 雨雾 / 遮挡、跨摄像头泛化、长视频稳定计数、端到端 P95 时延、真实事故率变化。逐条验证计划见 `docs/BOUNDARIES.md`。

### 3.4 硬件条件

本次演练：Windows 10 / AMD64 Family 23 Model 104 / 12 线程 CPU / 无 GPU / OpenCV 5.1.0 / NumPy 2.5.5。
正式训练与推理的 GPU 型号、CUDA / cuDNN、TensorRT 是否开启，须由本人在实验卡中如实填写。

---

## 4. 使用说明 / 维护

```
helmet-safety-yolo/
├── run.py                 统一入口
├── configs/               冻结配置（阈值、数据、A/B 条件）
├── src/                   系统基线与三个改动模块
│   ├── detector.py        YOLO / demo 双后端
│   ├── tiling.py          ① 重叠切片推理
│   ├── temporal.py        ② 跨帧时序投票
│   ├── alert.py           ③ 分级预警与事件闭环
│   ├── pipeline.py        成像→预处理→检测→判定→通讯/报表→业务动作
│   ├── metrics.py         P/R/mAP/FPS + 无真值保护
│   ├── synthetic.py       合成场景（SELFTEST_ONLY）
│   └── webapp/            实时预警看板
├── experiments/run_ab.py  公平 A/B 实验
├── scripts/               训练、数据集校验、实验卡生成
├── docs/                  架构、创新陈述、贡献、边界、证据索引
└── tests/                 核心逻辑单测
```

- **许可证**：MIT，见 `LICENSE`。
- **引用与数据集**：训练数据请使用可追溯来源并注明许可，见 `docs/DATASET.md`；本仓库不分发第三方数据集。
- **版本**：v1.0.0（2026-11-26）。
- **后续任务**：① 补真实逐帧真值并重跑 `MEASURED` 实验；② 用 TensorRT / 批处理把切片推理的 FPS 拉回可用区间；③ 引入 ByteTrack 替换简易 IoU 关联；④ 长视频稳定计数与 P95 时延报告；⑤ 现场联调日志。

---

## 5. 复现清单 · 课前诊断五问

| 五问 | 本仓库的回答位置 |
| --- | --- |
| 解决了什么真实问题？ | 本页 §1、`docs/INNOVATION.md` |
| 相对基线改了什么？ | `docs/ARCHITECTURE.md`（先画基线再圈改变）、`docs/INNOVATION.md` |
| 哪些实测数据支持？ | `experiments/results/*.json`、`docs/EXPERIMENT_CARD.md` |
| 本人负责哪些工作？ | `docs/CONTRIBUTION.md` |
| 陌生人怎样复现？ | 本页 §2、`requirements.txt`、`docs/EVIDENCE.md` |

> 课程要求：**能看、能跑、能对比、能复述**。本仓库四条都可在 5 分钟内验证。
