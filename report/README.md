# 正式实验报告

作者：庹力元，学号 59S2311；东南大学生命科学与技术学院，生物、计算机双学位学士。

- `main.tex`：正式中文报告源码。
- `main.pdf`：编译后的报告。
- `tables/`、`figures/`：从冻结实验记录生成的表格和插图。

2026-09-15 图文修订版包含 **17 幅图**：五种原方法架构，ContextFusion 与六模型集成结构，真实切片及多尺度输入，基因面板、验证曲线、残差相关、逐基因诊断、主结果和空间示例。正文逐图说明，图目录提供快速定位。

原方法图保留原论文参数，不代表本地适配完全复现作者配置。第三方图片出处、PDF 页码、裁切范围和文件摘要保存在 `figures/sources.json`；图片权利归原作者或出版方，不属于本仓库自有代码授权范围。GenAR 图已从论文重新完整提取，修正旧阶段 PPT 素材的下半部裁切问题。

在仓库根目录生成素材，然后编译：

```bash
python experiments/build_report_assets.py
python experiments/report_spatial_ablation.py
python experiments/expand_report_figures.py
cd report
latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex
```

`main.tex` 使用 TeX Live 自带 Fandol 字体，Windows/WSL 均可编译。源码和素材已随仓库提供；仅重新编译 PDF 不需要原始数据或训练权重。重建素材则需要本地完整实验预测。

补充实验固定 12 种预测空间平滑候选和原始输出，仅按 C1 HEG200 选择，所有平滑候选均未改善 C1，保留原六模型结果。记录位于 `../benchmarks/liver/report_spatial_ablation/`。此次未修改训练权重、基因面板或测试真值。数据的生物学来源（Andrews 等，2024）与 BLEEP 数据整理及预测方法已分别引用；同数据论文的结果仅作背景，不跨协议排名。
