# 评估报告（全部为实测）

> 由 `scripts/07_generate_docs.py` 生成。原始数据：
> `experiments/results/evaluation.json`、`benchmark.json`、`ab_summary.json`。

## 1. 官方验证（held-out `test` split）

| 指标 | 实测值 |
|---|---|
| Precision | 0.8387 |
| Recall | 0.7927 |
| mAP@0.5 | 0.7893 |
| mAP@0.5:0.95 | 0.4486 |
| 验证耗时 | 22.34 s |
| 测试图片数 | 600 |

### 选参集与独立测试集的落差（不隐藏）

`best.pt` 是按 **valid** split 的 fitness 选出的，因此 valid 指标带有选参乐观偏差；
下面把两者并列，而不是只报告更好看的那个数字。

| mAP@0.5 | 数值 | 说明 |
|---|---|---|
| valid（选 best.pt 用） | 0.8382 | 训练过程中逐 epoch 验证 |
| test（本报告） | 0.7893 | 与训练、选参完全隔离的独立测试集 |
| 差值 | 0.0489 | valid − test，正值即泛化落差 |

> 对外只引用 **test** 数字。valid 数字仅用于解释落差，不作为项目成绩。

## 2. 延迟基准

| 指标 | 实测值 |
|---|---|
| 设备 | cpu |
| 输入尺寸 | 320 |
| 预热 / 重复次数 | 3 / 20 |
| 均值 | 29.58 ms |
| P50 / P95 | 29.68 / 30.84 ms |
| 换算 FPS | 33.8 |

> Single-image synchronous inference on CPU; excludes video decode and post-processing.

## 3. 场景切片（诊断用，非 mAP）

方法：Class-aware greedy IoU@0.5 on raw predictions; slices by luminance and object density.

| 切片 | 图片数 | Precision | Recall | F1 | TP/FP/FN |
|---|---|---|---|---|---|
| brightness:bright | 47 | 0.7736 | 0.8119 | 0.7923 | 82/24/19 |
| brightness:dark | 50 | 0.7123 | 0.8387 | 0.7703 | 104/42/20 |
| brightness:normal | 503 | 0.7724 | 0.8104 | 0.7909 | 1154/340/270 |
| density:crowded | 93 | 0.8282 | 0.7773 | 0.8019 | 569/118/163 |
| density:few | 246 | 0.7300 | 0.8102 | 0.7680 | 538/199/126 |
| density:single | 261 | 0.7236 | 0.9209 | 0.8104 | 233/89/20 |

切片用于发现「哪类场景更差」，不等价于 COCO mAP，也不用于对外宣称精度。

## 4. 单变量公平 A/B

冻结条件：同一权重、同一批真实图片、同一顺序、同一 seed；每次仅改一个开关。
完整 JSON：`experiments/results/ab_*.json`。

| 实验 | A 基线 | B 变体 | 结论边界 |
|---|---|---|---|
| `E1_tiling` | R 0.8231 / 32.1 ms / 2 事件 | R 0.8027 / 291.3 ms / 6 事件 | 上游已把图片统一缩放到 640×640，本数据分布上切片未带来召回增益，反而增加约 9 倍时延与额外误检；因此 configs/inference/default.yaml 默认 enable_tiling=false，不把未证实的改进设为默认。 |
| `E2_confidence` | R 0.8231 / 30.1 ms / 2 事件 | R 0.6701 / 30.4 ms / 1 事件 | 阈值只改变判定门槛，不改模型权重；实测提高阈值后 Recall 下降、误报减少，是典型的 P/R 权衡，业务侧需按可接受误报率选择。 |
| `E3_temporal` | R 0.5952 / 30.2 ms / 1 事件 | R 0.7143 / 30.1 ms / 2 事件 | 受控扰动实验：每图构造 5 帧片段，对真实检出按固定 seed 注入 30% 瞬时漏检；两臂看到完全相同的帧序列，唯一差异是多数投票。收益=漏检恢复，代价=固定 3 帧确认延迟。两臂 wall-clock 时延差异处于测量噪声范围，不作为性能结论；片段由离散图片构造，不能外推真实连续视频流稳定性。 |
| `E4_dedup` | R 0.8231 / 33.1 ms / 2 事件 | R 0.8231 / 30.2 ms / 2 事件 | 本批 100 张真实图片上告警本就稀疏，冷却去重未触发，两臂检测指标与事件数完全一致；这只说明该策略在此样本量下无效果，不代表现场更安全或事故率下降，需更长连续序列验证。 |

## 5. 失败案例（误差最大的真实图片）

| 图片 | GT 框 | 预测框 | FP | FN | 亮度 |
|---|---|---|---|---|---|
| -746-_png_jpg.rf.967d39938cb4279bc5c949e7f8c0900d.jpg | 30 | 16 | 10 | 24 | 129.9 |
| Image21_jpg.rf.b45d1f5fd24dd10f31fc00e15c5d2eb6.jpg | 9 | 10 | 9 | 8 | 104.2 |
| 000162_jpg.rf.b072caeb7b615a99702f23b24396ea12.jpg | 12 | 5 | 4 | 11 | 128.9 |
| S1-EV360_FACEMA_M_20200130144349_jpg.rf.c26f7fa39f6790149edd67ff015b92fb.jpg | 15 | 22 | 11 | 4 | 81.8 |
| -3888-_png_jpg.rf.f82d4df0846a9d2ac3bd4b0f1dcf10eb.jpg | 2 | 13 | 12 | 1 | 85.0 |
| no-141-_png_jpg.rf.cc5c89ab756fd7fb53d01b4e83e95388.jpg | 2 | 9 | 9 | 2 | 131.4 |
| -1690-_png_jpg.rf.67eb652fb8cbd02ffa1249fed97a6374.jpg | 11 | 9 | 4 | 6 | 139.6 |
| -1699-_png_jpg.rf.7937cfd51475ed9141676c70c197d137.jpg | 4 | 7 | 6 | 3 | 107.0 |

可视化证据：`assets/real_inference.png`（绿框真值 / 红框预测）。

## 6. 结论边界

- Metrics are measured on a single public dataset, not on the customer's own camera distribution.
- CPU latency is not an edge-deployment benchmark; no TensorRT/ONNX optimization applied yet.
- Slice results use a simple greedy matcher and are diagnostic, not a replacement for official mAP.
