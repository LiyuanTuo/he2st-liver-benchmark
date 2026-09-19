# 实验编排

`optimize.py` 运行 `configs/` 中三个固定方案，保留所有候选，按 C1 选择后评价 D1。一般使用 `run.py fit` 完成单次训练；需要完整对照才运行本脚本。指标定义见 `docs/BENCHMARK.md`。

| 脚本 | 用途 | 所需输入 |
|---|---|---|
| `audit_preparation.py` | 重建数组与原缓存对比 | 本地源数据与原缓存 |
| `audit_repository.py` | 检查 Git 候选文件体积与大文件排除 | 本地 Git 仓库 |
| `verify_optimization.py` | 新权重重新推理、重算指标、空间块区间 | 本批完整实验产物 |
| `report_optimization.py` | 公开小型结果快照、图表和报告 | 本批及上一轮本地结果 |
| `export_prediction.py` | 将无RNA预测导出AnnData | 预测NPZ，可选H&E缩略图和比例文件 |
| `build_report_assets.py` | 生成正式LaTeX报告的表格和图 | 冻结结果及预测 |
| `expand_report_figures.py` | 生成数据、模型和诊断补充图 | 本地数据、论文图及实验结果 |
| `report_spatial_ablation.py` | 汇总空间平滑对照 | 本地预测及验证结果 |

报告脚本服务于本地研究记录，不是核心训练的依赖。AnnData 导出需要已有 `anndata`。最终 PPT 已归档于 `presentation/`；不再维护临时 PPT 更新工具。
