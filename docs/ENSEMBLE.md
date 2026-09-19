# 六模型集成

基础组包含三个随机种子的 ResNet18；扩展组包含 ResNet18、EfficientNet-B0 和训练标签平滑的 ResNet18。两组各自取预测均值，再按 0.75/0.25 加权，计算单位为固定 HEG200 的 log 相对表达。

| 组成模型 | 随机种子 | 最终权重 |
|---|---:|---:|
| 基础组 ResNet18 | 42 | 1/4 |
| 基础组 ResNet18 | 17 | 1/4 |
| 基础组 ResNet18 | 83 | 1/4 |
| 扩展组 ResNet18 | 42 | 1/12 |
| 扩展组 EfficientNet-B0 | 42 | 1/12 |
| 扩展组 ResNet18，20% 训练标签平滑 | 42 | 1/12 |

C1 先从扩展组单模型及等权平均中选出三模型平均，再比较扩展组占比 0、25%、50%、75%、100% 的组合。25% 的 C1 HEG200 PCC 最高，为 0.3665。三份扩展组权重均选中当前参数，未选中 EMA。

[集成结构图](../report/figures/ensemble_method.pdf) · [权重路径与配置](../benchmarks/liver/final_ensemble.json) · [模型选择记录](../benchmarks/liver/optimization_selection_locked.json)

## D1 结果

| 模型 | HEG200 PCC | HEG50 PCC | MAE |
|---|---:|---:|---:|
| ST-Net 适配 | 0.1472 | 0.1978 | 0.4378 |
| 基础组三模型平均 | 0.2302 | 0.3156 | 0.4256 |
| 六模型集成 | 0.2343 | 0.3180 | 0.4274 |
| 全随机初始化单模型 | 0.2302 | 0.3278 | 0.4197 |

相对基础组，六模型的 PCC 增加 0.0042/0.0024，MAE 增加 0.0018。按 D1 的 24 个空间块进行 300 次配对 bootstrap，HEG200 增益的 95% 区间为 [0.0009, 0.0077]，HEG50 为 [-0.0037, 0.0091]。区间描述单张切片内的抽样变化。

六个模型的展平残差相关系数为 0.974–0.992，预测误差高度相关。完整结果见[基线表](../benchmarks/liver/baseline_20260913.csv)、[组合比较](../benchmarks/liver/optimization_20260914.csv)和[残差分析](../benchmarks/liver/ensemble_error_analysis.json)。

## 推理

准备三尺度图像缓存及配置列出的六份权重，在仓库根目录运行：

```bash
python run.py ensemble --slides C73_D1 --output results/final_prediction.npz
python run.py evaluate --prediction results/final_prediction.npz --truth data/processed/gse240429_heg/arrays/C73_D1.npz --output results/final_metrics.json
```

程序先校验权重的 SHA256，再按各自的旋转增强设置生成预测。2026-09-15 的重推理与保存的 2265×200 预测矩阵逐元素一致，见[核验记录](../benchmarks/liver/final_ensemble_verification.json)。训练步骤见[运行指南](QUICKSTART.md)。
