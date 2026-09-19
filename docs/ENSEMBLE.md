# 最终集成：方法、效果和复现

**最终预测 = 0.75 × 原模型组均值 + 0.25 × 新模型组均值。** 所有运算在同一固定 HEG200 的预测 log 表达空间进行，不平均 PCC 分数，不读取测试 RNA。

| 组成模型 | 最终权重 |
|---|---:|
| 原 ContextFusion ResNet18，seed42 | 1/4 |
| 原 ContextFusion ResNet18，seed17 | 1/4 |
| 原 ContextFusion ResNet18，seed83 | 1/4 |
| 新 ResNet18 配方，seed42 | 1/12 |
| 新 EfficientNet-B0 配方，seed42 | 1/12 |
| 新 ResNet18，训练标签20%邻域平滑，seed42 | 1/12 |

这是多种子与不同训练方案的预测平均，不是基于 bootstrap 样本重训的 bagging，也不是训练 stacking 元模型。C1 先从新单模型及均值中选中三模型均值，再比较原组、新组及新组占25%/50%/75%的组合，以 C1 HEG200 PCC 锁定25%。三组新模型最终均选择原权重，EMA未胜出。

[集成流程图（PDF）](../report/figures/ensemble_method.pdf)

| 比较 | D1 HEG200 | D1 HEG50 | MAE |
|---|---:|---:|---:|
| 原 ST-Net | 0.1472 | 0.1978 | 0.4378 |
| 原三种子集成 | 0.2302 | 0.3156 | 0.4256 |
| 最终六模型集成 | **0.2343** | **0.3180** | 0.4274 |
| 前轮全随机单模型 | 0.2302 | 0.3278 | 0.4197 |

最终集成相对原三种子只有小幅增益，MAE略变差，且HEG50没有超过全随机单模型。24个空间块、300次配对bootstrap的HEG200增益区间为[0.0009,0.0077]，HEG50为[-0.0037,0.0091]。这只描述当前切片，不代表独立患者泛化。

六个模型的展平残差相关性约0.974–0.992，说明误差高度共享，能够通过平均消除的差异有限；共同测量噪声、基因尺度和偏差也会影响此统计量，不能从它单独确定误差原因。诊断在锁定结果后进行，未用于选权重。

## 从六份权重直接推理

```bash
python run.py ensemble --slides C73_D1 --output results/final_prediction.npz
python run.py evaluate --prediction results/final_prediction.npz --truth data/processed/gse240429_heg/arrays/C73_D1.npz --output results/final_metrics.json
```

[冻结配置](../benchmarks/liver/final_ensemble.json)记录六份权重的路径、SHA256、epoch、TTA和分组。执行前会核对全部权重，缺失时不会下载或跳过某个成员。准备好的图像缓存及六份权重需留在本地；Git仓库不直接分发这些大文件。各组训练及历史结果复现见[运行指南](QUICKSTART.md)。

2026-09-15直接加载六份权重，对2265×200表达值重推理，结果与冻结预测逐元素一致，最大差为0，见[独立集成核验](../benchmarks/liver/final_ensemble_verification.json)。

正式推导、完整结果、误差分析和实验限制见[LaTeX实验报告](../report/main.pdf)。
