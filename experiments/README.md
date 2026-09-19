# 实验脚本

在仓库根目录运行。单次训练使用 `run.py fit`，多配置比较使用 `experiments/optimize.py`。

| 脚本 | 用途 | 输入 |
|---|---|---|
| `optimize.py` | 运行三组配置，按 C1 选择后评价 D1 | 数据缓存、训练配置 |
| `audit_preparation.py` | 对比重建数组与原缓存 | 源数据、缓存 |
| `audit_repository.py` | 检查待提交文件类型和体积 | Git 文件列表 |
| `verify_optimization.py` | 重推理、重算指标和空间块区间 | 权重、预测和真值 |
| `report_optimization.py` | 汇总结果和图表 | 基础组及扩展组结果 |
| `export_prediction.py` | 导出 AnnData | 预测 NPZ，可选图像元数据 |
| `build_report_assets.py` | 生成报告表格和插图 | 保存的结果及预测 |
| `expand_report_figures.py` | 生成数据、架构和诊断图 | 数据、论文图和实验结果 |
| `report_spatial_ablation.py` | 比较预测空间平滑 | 保存的集成预测及真值 |

AnnData 导出需要 `anndata`；报告制图需要 `pyproject.toml` 中的 `reports` 依赖。实验协议见[基准说明](../docs/BENCHMARK.md)。
