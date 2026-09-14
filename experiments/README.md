# 实验编排

`optimize.py` 运行 `configs/` 中三个固定方案，保留所有候选，按 C1 选择后评价 D1。一般使用 `run.py fit` 完成单次训练；需要完整对照才运行本脚本。指标定义见 `docs/BENCHMARK.md`。

| 脚本 | 用途 | 所需输入 |
|---|---|---|
| `audit_preparation.py` | 重建数组与原缓存对比 | 本地源数据与原缓存 |
| `audit_repository.py` | 检查 Git 候选文件体积与大文件排除 | 本地 Git 仓库 |
| `verify_optimization.py` | 新权重重新推理、重算指标、空间块区间 | 本批完整实验产物 |
| `report_optimization.py` | 公开小型结果快照、图表和报告 | 本批及上一轮本地结果 |
| `update_optimization_ppt.py` | 就地更新原11页PPT | 本地原PPT、历史样式脚本及结果 |
| `export_prediction.py` | 将无RNA预测导出AnnData | 预测NPZ，可选H&E缩略图和比例文件 |
| `build_report_assets.py` | 生成正式LaTeX报告的表格和图 | 冻结结果及预测 |
| `finalize_presentation.py` | 完善作者、集成方法及最终预测图 | 原11页PPT及报告图 |

报告/PPT脚本服务于本地研究记录，不是核心训练的依赖。AnnData 导出需要已有 `anndata`；PPT/PDF渲染使用本机 PowerPoint。
