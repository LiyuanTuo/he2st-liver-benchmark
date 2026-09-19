# H&E 图像预测空间基因表达

本项目使用 GSE240429 的四张人肝连续切片，比较 ResSAT、GenAR、Stem、ST-Net 和 BLEEP 的适配实现，并训练三尺度 ContextFusion 模型。

A1、B1 用于训练，C1 用于验证，D1 用于测试。训练集选取 200 个高表达基因（HEG200），其中表达最高的 50 个组成 HEG50。六模型集成在 D1 上的平均 PCC 分别为 **0.2343、0.3180**，ST-Net 适配为 0.1472、0.1978。集成组成及完整比较见[集成方法](docs/ENSEMBLE.md)和[实验报告源码](report/main.tex)。

## 运行

需要 Python 3.11 及以上、PyTorch 和三尺度图像缓存。依赖见 [pyproject.toml](pyproject.toml)，数据准备见 [data/README.md](data/README.md)。在仓库根目录执行：

```bash
python run.py doctor
python run.py fit --config configs/resnet18_ema.json --output results/my_run
python run.py predict --checkpoint results/my_run/best.pt --slides C73_D1 --output results/my_run/D1_prediction.npz
python run.py evaluate --prediction results/my_run/D1_prediction.npz --truth data/processed/gse240429_heg/arrays/C73_D1.npz --output results/my_run/D1_metrics.json
```

以上命令训练和评价单模型。六模型推理见[运行指南](docs/QUICKSTART.md)。原始数据、图像缓存和训练权重需单独准备，不随仓库分发。

## 目录

| 路径 | 内容 |
|---|---|
| `src/he2st/`、`run.py` | 数据处理、训练、预测与评价 |
| `configs/` | 三组训练配置 |
| `benchmarks/liver/` | 基因列表、实验配置与结果 |
| `experiments/` | 批量实验、结果核验和制图 |
| `scripts/` | 各方法适配与早期实验脚本 |
| `docs/` | 使用说明、方法与诊断记录 |
| `report/`、`presentation/` | LaTeX 报告及演示材料 |
| `data/`、`results/` | 本地数据和训练输出 |

[基准协议](docs/BENCHMARK.md) · [模型结构](docs/MODEL.md) · [论文来源](papers/README.md) · [文档索引](docs/README.md)

测试命令：`python -m unittest discover -s tests -v`。
