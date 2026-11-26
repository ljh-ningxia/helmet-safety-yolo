# 系统基线与改动位置

> 页面唯一结论：**先理解原系统，才能知道改动到底解决了什么。**

## 1. 完整基线流程（改动前）

```
成像 → 预处理 → 定位/检测 → 判定 → 通讯/报表 → 业务动作
 │        │          │         │         │          │
摄像头   限幅/归一   YOLO     单帧阈值   写日志    现场提示
/RTSP    A/B 一致   整图推理  即判即报   画框      （常常缺失）
```

支撑知识：相机 / 镜头 / 光源 · OpenCV 图像处理 · YOLOv8 · 事件驱动 · Web 看板。

## 2. 圈出本人改动模块

| 环节 | 基线 A | 改进 B | 对应创新 |
| --- | --- | --- | --- |
| 成像 | 摄像头 / RTSP / 图片 / 视频 | **不变** | — |
| 预处理 | 仅等比限幅（1920 上限） | **不变** | — |
| 定位/检测 | 整图一次 `predict` | 重叠切片 + 放大 + 跨切片 NMS | ① `src/tiling.py` |
| 判定 | 单帧 `cls==1` 即报 | IoU 跨帧关联 + 连续 K 帧投票 | ② `src/temporal.py` |
| 通讯/报表 | 仅内存对象 | 分级事件 + JSONL 台账 + 证据快照 | ③ `src/alert.py` |
| 业务动作 | 画框（无处置） | L1/L2/L3 预警 + 冷却去重 + 可选 Webhook | ③ `src/alert.py` |

## 3. 为什么必须分开改

如果一次性把三处都打开，就**无法回答「是哪一处带来了变化」**。因此 `PipelineConfig.variant()` 只接受 4 种派生：

```
A            → 三个开关全关
B_tiling     → 只开 enable_tiling
B_temporal   → 只开 enable_temporal
B_dedup      → 只开 enable_alert_dedup
```

`experiments/run_ab.py` 的每组实验都从同一 `base_cfg` 派生 A 与 B，并把冻结条件、环境指纹、检测器描述一起写进 JSON，供陌生人复核。

## 4. 系统分层

| 层 | 文件 | 职责 |
| --- | --- | --- |
| 入口 | `run.py` | selftest / ab / detect / web / synthetic / report |
| 数据源 | `src/datasource.py` | 合成 / 图片 / 视频 / 摄像头 / 数据集目录；是否带真值显式声明 |
| 检测 | `src/detector.py` | YOLO 正式后端 + demo 演练后端，绝不静默降级 |
| 基线 | `src/pipeline.py` | 串起六个环节，三个开关 |
| 创新 | `tiling.py` / `temporal.py` / `alert.py` | 三项独立改动 |
| 评估 | `src/metrics.py` | TP/FP/FN、P/R/mAP、FPS、失败率；无真值保护 |
| 展示 | `src/webapp/` | 实时画面、告警列表、指标卡、A/B 表、证据等级水印 |
| 证据 | `experiments/` | 实验脚本、结果 JSON、对比图、实验卡 |

## 5. 工程权衡（必须同时报告）

| 优化收益 | 可能付出的成本 |
| --- | --- |
| 远景小目标召回上升 | 推理次数 ×N、FPS 下降、误检增多、mAP 可能下降 |
| 告警时序稳定 | K-1 帧固有延迟、快速通过目标可能漏报 |
| 事件可追溯、不刷屏 | 磁盘 / 网络开销、冷却窗口内同一目标不再上报 |

> 详见 `docs/BOUNDARIES.md` 与 `docs/EXPERIMENT_CARD.md`。
