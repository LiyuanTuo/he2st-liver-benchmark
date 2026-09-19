# 运行指南

## 环境与数据

需要 Python 3.11 及以上，依赖列于 [pyproject.toml](../pyproject.toml)。已有环境可直接使用；缺少依赖时运行 `python -m pip install -r requirements.txt`。GPU 训练需要支持 CUDA 的 PyTorch。

在仓库根目录执行。Windows 可用 `py -3.12`，WSL 可用 `python3`：

```bash
python run.py doctor
python -m unittest discover -s tests -v
```

按[数据说明](../data/README.md)准备表达数组、三尺度图像缓存和基因列表。

## 单模型

```bash
python run.py fit --config configs/resnet18_ema.json --output results/my_run
python run.py predict --checkpoint results/my_run/best.pt --slides C73_D1 --output results/my_run/D1_prediction.npz
python run.py evaluate --prediction results/my_run/D1_prediction.npz --truth data/processed/gse240429_heg/arrays/C73_D1.npz --output results/my_run/D1_metrics.json
```

`fit` 使用训练和验证表达；`predict` 使用图像、基因名、条形码和坐标；`evaluate` 读取实测表达，核对样本与基因顺序后计算指标。CPU 运行可加 `--device cpu`。

每次训练保存 `protocol.json`、`history.json`、`best.pt`、`selection.json` 和 `validation.npz`。保留权重旁的 `selection.json`；更换配置时使用新的输出目录。

## 多配置实验

```bash
python experiments/optimize.py --output results/my_sweep
```

该命令运行 `configs/` 中的三组配置，按 C1 HEG200 PCC 选择权重、旋转增强和组合比例，保存 `selection_locked.json` 后评价 D1。

报告中的六模型结果还包含基础组三个随机种子的预测。已有这些预测时，运行：

```bash
python experiments/optimize.py --output results/optimization_20260914 --baseline-validation results/liver_context_ensemble_20260913/selected_C1.npz --baseline-test results/liver_context_ensemble_20260913/selected_D1.npz
```

基础组的训练脚本为 `scripts/59_train_context_fusion.py`、`scripts/64_context_seed_repeats.py`，预测平均由 `scripts/65_context_ensemble.py` 完成。配置及结果见[多尺度实验](06_LIVER_CONTEXT_IMPROVEMENT.md)。

## 六模型推理

准备[集成配置](../benchmarks/liver/final_ensemble.json)列出的六份权重后运行：

```bash
python run.py ensemble --slides C73_D1 --output results/final_prediction.npz
python run.py evaluate --prediction results/final_prediction.npz --truth data/processed/gse240429_heg/arrays/C73_D1.npz --output results/final_metrics.json
```

权重和图像缓存保存在本地。集成系数、文件校验和结果见[集成说明](ENSEMBLE.md)。

## 报告编译

LaTeX 源码和插图位于 `report/`，使用 XeLaTeX 编译：

```bash
latexmk -cd -xelatex -interaction=nonstopmode -halt-on-error report/main.tex
```
