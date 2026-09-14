# 数据准备

数据文件不随 Git 仓库提交。固定划分和面板见 [基准协议](../docs/BENCHMARK.md)。

## 已有本地数据

本项目原缓存可直接运行新 `run.py`，不必重建。需要的文件为：

```text
processed/gse240429_heg/arrays/C73_{A1,B1,C1,D1}.npz
processed/gse240429/context_20260913/C73_{A1,B1,C1,D1}_{224,896,1792}_patches.npy
```

NPZ 必须有同序 `genes`、`barcodes`、全分辨率 `coordinates_xy`；训练/评价还需要 `raw_counts`。预测模式可以完全不提供 RNA。图像数组是 `[spots,224,224,3]` 的 uint8 RGB。

## 从源文件重建

表达和坐标来自 [BLEEP 官方数据目录](https://github.com/bowang-lab/BLEEP/tree/main/GSE240429_data)，参考提交 `2395967`。下载该仓库的数据目录后，源目录应包含：

```text
filtered_expression_matrices/{1,2,3,4}/{matrix.mtx,features.tsv,barcodes.tsv}
tissue_pos_matrices/tissue_positions_list_{1,2,3,4}.csv
```

全分辨率 H&E 来自 [GSE240429](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE240429) 的 GSM7697868–GSM7697871；对应 A1–D1。下载各样本 `GEX_C73_*_Merged.tiff.gz`，完整解压为可内存映射的 RGB TIFF，分别命名 `C73_A1.tif` 等，放在 `processed/gse240429/tiff/`。不能用 hires 缩略图替代全分辨率 TIFF，否则 spot 坐标比例错误。历史下载与完整性校验脚本见 `scripts/README.md`。

```bash
python run.py prepare --source third_party/BLEEP/GSE240429_data/data
```

命令先用 A1+B1 重新验证固定 HEG200/50 排名，再按公开面板列序写表达数组和三尺度图像缓存，不需要运行 GenAR。已有输出会拒绝覆盖；验证重建可指定 `--arrays results/preparation_audit --arrays-only`，或通过 `--arrays`、`--images` 指定全新目录。自定义输出需同步修改训练配置中的路径。

`prepare` 不会自动下载几十 GB 的数据，也不会重新安装环境。原始 TIFF、压缩包和缓存均保留在本地，发布代码时由 `.gitignore` 排除。
