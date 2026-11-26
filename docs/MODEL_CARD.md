# Model Card — YOLOv8n Hard Hat Detector

> 由 `scripts/07_generate_docs.py` 从 `train_run.json` / `evaluation.json` 生成。

## 模型概要

| 项 | 值 |
|---|---|
| 架构 | YOLOv8n (Ultralytics 8.4.x) |
| 基座权重 | `models/yolov8n.pt`，SHA256 `f59b3d833e2ff32e…` |
| 基座许可 | AGPL-3.0（见 `NOTICE.md`，商用需另取授权） |
| 训练产物 | `experiments/runs/yolov8n-hardhat/weights/best.pt` |
| 类别 | `hardhat`, `no-hardhat` |
| 用途 | 工地图像/视频帧中的安全帽佩戴状态检测与未佩戴告警 |

## 训练配置（实测运行记录）

| 参数 | 值 |
|---|---|
| model | models/yolov8n.pt |
| epochs | 12 |
| imgsz | 320 |
| batch | 16 |
| device | cpu |
| workers | 0 |
| seed | 20261126 |
| deterministic | True |
| lr0 | 0.01 |
| lrf | 0.01 |
| weight_decay | 0.0005 |
| patience | 10 |
| amp | False |
| mosaic | 0.5 |
| data_yaml | data/processed/data.yaml |

- 训练耗时：`2499.94 s`，记录 epoch 数：`12`
- 环境：Python `3.13.14`、torch `2.15.0+cpu`、
  ultralytics `8.4.219`、设备 `cpu`、
  CPU `12` 核、git `349a4073`
- 完整可复现命令：`python3.exe scripts/03_train.py`
- 逐 epoch 曲线：`experiments/runs/yolov8n-hardhat/results.csv`

## 独立测试集性能（MEASURED）

| 类别 | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|
| hardhat | 0.8574 | 0.7560 | 0.7630 | 0.3861 |
| no-hardhat | 0.8200 | 0.8295 | 0.8155 | 0.5112 |

| 总体指标 | 值 |
|---|---|
| Precision | 0.8387 |
| Recall | 0.7927 |
| mAP@0.5 | 0.7893 |
| mAP@0.5:0.95 | 0.4486 |
| 测试图片数 | 600 |
| 推理均值时延 | 29.58 ms |
| 推理 P95 时延 | 30.84 ms |
| 吞吐（由均值换算） | 33.8 FPS |

> 评估 split：`test`，conf `0.25`，IoU `0.6`。
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
