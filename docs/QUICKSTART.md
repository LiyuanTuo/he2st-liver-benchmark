# 运行指南

在仓库根目录执行。已有 Python 环境可直接运行，不需要创建虚拟环境或重新下载 PyTorch。

```bash
python run.py doctor
python -m unittest discover -s tests -v
```

Windows 可以用 `py -3.12` 替换 `python`；GPU 实验使用已安装 CUDA PyTorch 的 WSL 全局 `python3`。数据准备见 [数据目录](../data/README.md)。当前配置需要三尺度图像缓存和固定基因面板。

## 单模型训练、预测与评价

```bash
python run.py fit --config configs/resnet18_ema.json --output results/my_run
python run.py predict --checkpoint results/my_run/best.pt --slides C73_D1 --output results/my_run/D1_prediction.npz
python run.py evaluate --prediction results/my_run/D1_prediction.npz --truth data/processed/gse240429_heg/arrays/C73_D1.npz --output results/my_run/D1_metrics.json
```

`fit` 只读训练和验证 RNA；`predict` 只读图像、基因名称、barcode 和坐标，输出不包含实测 RNA。`evaluate` 单独读取真值，并检查基因、barcode、坐标完全一致。无 GPU 时可对 `fit`、`predict` 加 `--device cpu`，完整训练会很慢。

训练输出包含 `protocol.json`、`history.json`、`best.pt`、`selection.json`、`validation.npz`。同配置的完整训练可复用；更改配置必须使用新输出目录。权重与相邻的 `selection.json` 必须一起保存。

## 三组优化对照

```bash
python experiments/optimize.py --output results/my_sweep
```

依次运行三个配置，在 C1 选择原权重或 EMA、TTA 与模型组合；写入 `selection_locked.json` 后才评价 D1。所有候选均保留。已有上一轮集成预测时，可额外传入 `--baseline-validation` 和 `--baseline-test`，加入事先声明的混合候选；默认不依赖这些本地历史文件。

本轮报告的 0.2343 / 0.3180 是 75% 上一轮三种子集成 + 25% 本轮三个新模型均值，不是单个 ResNet18 的成绩。本地原实验命令为：

```bash
python experiments/optimize.py --output results/optimization_20260914 --baseline-validation results/liver_context_ensemble_20260913/selected_C1.npz --baseline-test results/liver_context_ensemble_20260913/selected_D1.npz
```

上一轮产物不随 Git 分发，其训练与集成过程见历史脚本 `59_*`、`63_*`、`65_*` 及 `docs/06_LIVER_CONTEXT_IMPROVEMENT.md`。新 checkout 未准备历史产物时，使用上面的默认三组命令；应比较报告中的新模型行，不能声称复现了最终六模型组合。

## 安装与发布边界

`run.py` 可直接使用源码，无需安装本项目。确需安装命令行时，在已有依赖环境运行 `python -m pip install -e . --no-deps`；缺失依赖见 `pyproject.toml`。`requirements.txt` 是安装入口，`requirement.txt` 是本地任务记录，两者不同。

`data/`、`results/`、`third_party/`、论文、权重和历史 PPT 默认不进入 Git。发布结果放在 `benchmarks/liver/`；正式报告及最终演示副本分别位于 `report/`、`presentation/`，由明确的白名单进入 Git。历史方法需要各自的第三方代码和额外依赖，见 [历史脚本索引](../scripts/README.md)。最终集成的直接推理见 [集成指南](ENSEMBLE.md)。
