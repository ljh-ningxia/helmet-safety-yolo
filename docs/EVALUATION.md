# 评估报告（全部为实测）

> 由 `scripts/07_generate_docs.py` 生成。原始数据：
> `experiments/results/evaluation.json`、`benchmark.json`、`ab_summary.json`。

## 1. 官方验证（held-out `test` split）

| 指标 | 实测值 |
|---|---|
| Precision | — |
| Recall | — |
| mAP@0.5 | — |
| mAP@0.5:0.95 | — |
| 验证耗时 | — s |
| 测试图片数 | — |

## 2. 延迟基准

| 指标 | 实测值 |
|---|---|
| 设备 | — |
| 输入尺寸 | — |
| 预热 / 重复次数 | — / — |
| 均值 | — ms |
| P50 / P95 | — / — ms |
| 换算 FPS | — |

> 

## 3. 场景切片（诊断用，非 mAP）

方法：—

| 切片 | 图片数 | Precision | Recall | F1 | TP/FP/FN |
|---|---|---|---|---|---|

切片用于发现「哪类场景更差」，不等价于 COCO mAP，也不用于对外宣称精度。

## 4. 单变量公平 A/B

冻结条件：同一权重、同一批真实图片、同一顺序、同一 seed；每次仅改一个开关。
完整 JSON：`experiments/results/ab_*.json`。

| 实验 | A 基线 | B 变体 | 结论边界 |
|---|---|---|---|

## 5. 失败案例（误差最大的真实图片）

| 图片 | GT 框 | 预测框 | FP | FN | 亮度 |
|---|---|---|---|---|---|

可视化证据：`assets/real_inference.png`（绿框真值 / 红框预测）。

## 6. 结论边界

- 见 docs/BOUNDARIES.md
