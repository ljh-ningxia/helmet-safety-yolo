# Model Card — YOLOv8n Hard Hat Detector

> 由 `scripts/07_generate_docs.py` 从 `train_run.json` / `evaluation.json` 生成。

## 模型概要

| 项 | 值 |
|---|---|
| 架构 | YOLOv8n (Ultralytics 8.4.x) |
| 基座权重 | `models/yolov8n.pt`，SHA256 `f59b3d833e2ff32e…` |
| 基座许可 | AGPL-3.0（见 `NOTICE.md`，商用需另取授权） |
| 训练产物 | `—` |
| 类别 | `hardhat`, `no-hardhat` |
| 用途 | 工地图像/视频帧中的安全帽佩戴状态检测与未佩戴告警 |

## 训练配置（实测运行记录）

| 参数 | 值 |
|---|---|

- 训练耗时：`— s`，记录 epoch 数：`—`
- 环境：Python `—`、torch `—`、
  ultralytics `—`、设备 `—`、
  CPU `—` 核、git `—`
- 完整可复现命令：`—`
- 逐 epoch 曲线：`—/results.csv`

## 独立测试集性能（MEASURED）

_按类指标未生成_

| 总体指标 | 值 |
|---|---|
| Precision | — |
| Recall | — |
| mAP@0.5 | — |
| mAP@0.5:0.95 | — |
| 测试图片数 | — |
| 推理均值时延 | — ms |
| 推理 P95 时延 | — ms |
| 吞吐（由均值换算） | — FPS |

> 评估 split：`test`，conf `—`，IoU `—`。
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
