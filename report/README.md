# 正式实验报告

作者：庹力元，学号 59S2311；东南大学生命科学与技术学院，生物、计算机双学位学士。

- `main.tex`：正式中文报告源码。
- `main.pdf`：编译后的报告。
- `tables/`、`figures/`：从冻结实验记录生成的表格和插图。

在仓库根目录生成素材，然后编译：

```bash
python experiments/build_report_assets.py
cd report
latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex
```

`main.tex` 使用 TeX Live 自带 Fandol 字体，Windows/WSL 均可编译。源码和素材已随仓库提供；仅重新编译 PDF 不需要原始数据或训练权重。重建素材则需要本地完整实验预测。
