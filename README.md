# H&E → 空间基因表达预测

在固定的人肝 GSE240429 基准上，比较 ResSAT、GenAR、Stem、ST-Net、BLEEP 等适配方法，并训练自建三尺度 ContextFusion 模型。

**固定协议：A1+B1 训练、C1 验证、D1 测试；训练集选定 HEG200 / HEG50；测试真值不做重建。** 本轮 C1 选择模型在固定 D1 的 PCC 为 **0.2343 / 0.3180**，原 ST-Net 为 **0.1472 / 0.1978**。完整对照和局限见[结果报告](docs/RESULTS.md)；固定面板平均仍未达到 0.5。

最终方案是 **75% 原三种子均值 + 25% 新三模型均值**。相对原集成只增加 0.0042 / 0.0024，MAE 略变差；HEG50 增益区间跨零。详见[集成方法与记录](docs/ENSEMBLE.md)。

**正式交付：** [实验报告 PDF](report/main.pdf) · [LaTeX 源码](report/main.tex) · [最终 PPTX（11页）](presentation/final_report.pptx) · [演示 PDF](presentation/final_report.pdf)。

报告现含 **17 幅图**，覆盖五种方法及自建网络架构、集成机制、真实输入和结果诊断，逐图附解释与来源。新增预测空间平滑对照未改善 C1，仍保留上述六模型结果；[修订与重建说明](report/README.md)。

## 快速开始

使用已有的全局 Python 环境，在仓库根目录运行：

```bash
python run.py doctor
python run.py fit --config configs/resnet18_ema.json --output results/my_run
python run.py predict --checkpoint results/my_run/best.pt --slides C73_D1 --output results/my_run/D1_prediction.npz
python run.py evaluate --prediction results/my_run/D1_prediction.npz --truth data/processed/gse240429_heg/arrays/C73_D1.npz --output results/my_run/D1_metrics.json
```

首次运行先按[数据说明](data/README.md)准备数据。Windows 可用 `py -3.12`，WSL 可用已有 CUDA 环境的 `python3`。不需要重新创建虚拟环境。

上面的命令训练单个 ResNet18；首页成绩来自 C1 选定的新旧模型组合，复现组合见[完整实验指南](docs/QUICKSTART.md)。

## 仓库结构

```text
run.py                  统一命令行入口
src/he2st/              数据准备、模型、训练、预测和指标
configs/                三组可复用训练配置
benchmarks/liver/       固定基因面板、协议与公开结果快照
experiments/            多配置实验编排与审计
tests/                  科学指标和数据边界测试
docs/                   运行指南、实验结果和历史记录
scripts/                编号历史脚本，保留复现路径
data/、results/         本地数据和实验产物，默认不进入 Git
```

## 文档

- [运行指南](docs/QUICKSTART.md)：训练、推理、完整对照和环境。
- [基准协议](docs/BENCHMARK.md)：高表达基因、归一化和 PCC 口径。
- [模型设计](docs/MODEL.md)：三尺度输入及 EMA、平滑监督等对照。
- [结果与原因](docs/RESULTS.md)：全部方案的实测成绩及局限。
- [论文与代码来源](papers/README.md) · [历史实验脚本](scripts/README.md)。

运行测试：`python -m unittest discover -s tests -v`。

这是单供体的跨切片研究，输出是图像预测的相对表达。不同论文的数据集、基因面板和重建目标不可直接比较。原始数据、第三方仓库和训练权重保留本地；公开仓库包含代码、冻结结果、正式报告及最终演示材料。
