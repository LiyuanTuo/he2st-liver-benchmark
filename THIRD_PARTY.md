# 外部代码与数据

`src/he2st/` 为本项目实现，CNN 编码器来自 torchvision。各方法适配参考以下作者仓库：

| 方法 | 仓库 | 参考提交 |
|---|---|---|
| ResSAT | [Wonderangela123/ResSAT](https://github.com/Wonderangela123/ResSAT) | `10d60ab` |
| GenAR | [oyjr/genar](https://github.com/oyjr/genar) | `223a0a6` |
| Stem | [SichenZhu/Stem](https://github.com/SichenZhu/Stem) | `cbc3c7c` |
| BLEEP | [bowang-lab/BLEEP](https://github.com/bowang-lab/BLEEP) | `2395967` |
| ST-Net | [bryanhe/ST-Net](https://github.com/bryanhe/ST-Net) | `43022c1` |
| HEST | [mahmoodlab/HEST](https://github.com/mahmoodlab/HEST) | `3ddb5ea` |

本地副本位于 `third_party/`，不随仓库分发。代码、数据和模型权重分别遵循原来源的使用条件。ImageNet 权重由 torchvision 缓存管理。

人肝图像来自 GSE240429，表达矩阵和坐标采用 BLEEP 整理版本。论文及数据链接见[来源清单](papers/README.md)。报告中的原论文架构图出处记录于 [sources.json](report/figures/sources.json)，图片权利归原作者或出版方。
