# H&E 图像预测空间基因表达

使用 GSE240429 的四张人肝连续切片，比较 ResSAT、GenAR、Stem、ST-Net、BLEEP，并训练三尺度 ContextFusion 模型。A1、B1 用于训练，C1 用于验证，D1 用于测试。

从训练集选取 200 个高表达基因（HEG200），其中表达最高的 50 个为 HEG50。测试结果如下：

| 方法 | HEG200 PCC | HEG50 PCC |
|---|---:|---:|
| ST-Net 适配 | 0.1472 | 0.1978 |
| ContextFusion 六模型集成 | 0.2343 | 0.3180 |

[方法与结果](docs/ENSEMBLE.md) · [实验报告源码](report/main.tex)

## 使用

需要 Python 3.11 及以上和 PyTorch，依赖见 [pyproject.toml](pyproject.toml)。原始数据、图像缓存和权重需按[数据说明](data/README.md)单独准备。

在仓库根目录运行环境检查和单模型训练：

```bash
python run.py doctor
python run.py fit --config configs/resnet18_ema.json --output results/my_run
```

预测、评价、集成和报告编译见[运行指南](docs/QUICKSTART.md)。

## 目录

| 路径 | 内容 |
|---|---|
| `src/he2st/`、`run.py` | 数据处理、训练、预测与评价 |
| `configs/` | 训练配置 |
| `benchmarks/liver/` | 基因列表与实验结果 |
| `experiments/` | 批量实验、核验和制图 |
| `scripts/` | 方法适配与早期实验 |
| `docs/` | 使用说明与实验记录 |
| `report/`、`presentation/` | LaTeX 报告与演示材料 |
| `data/`、`results/` | 本地数据和训练输出 |

[基准协议](docs/BENCHMARK.md) · [模型结构](docs/MODEL.md) · [论文来源](papers/README.md) · [全部文档](docs/README.md)

测试：`python -m unittest discover -s tests -v`。
