# 论文、数据与外部实现

核心 `src/he2st/` 是本项目的训练/预测实现，CNN 来自 torchvision。历史实验引用以下作者仓库；本地 `third_party/` 不会随本项目 Git 提交，也不改变原作者许可。

| 方法 | 作者代码 | 本地参考提交 |
|---|---|---|
| ResSAT | https://github.com/Wonderangela123/ResSAT | `10d60ab` |
| GenAR | https://github.com/oyjr/genar | `223a0a6` |
| Stem | https://github.com/SichenZhu/Stem | `cbc3c7c` |
| BLEEP | https://github.com/bowang-lab/BLEEP | `2395967` |
| ST-Net | https://github.com/bryanhe/ST-Net | `43022c1` |
| HEST | https://github.com/mahmoodlab/HEST | `3ddb5ea` |

完整论文出处、适配限制与本地完整性记录见 [论文清单](papers/README.md)。统一人肝结果是本项目的适配结果，不能替代原论文在其原始数据/特征/设备上的成绩。ImageNet 初始化权重由 torchvision 下载至用户已有缓存，不放进代码仓库。

数据来自 GSE240429 与 BLEEP 发布的表达/坐标文件。原始图像、RNA、论文 PDF 和第三方权重保持各自来源及使用条件；本仓库不重新分发这些大文件。
