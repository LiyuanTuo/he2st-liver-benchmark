# 数据准备

数据划分与基因选择见[基准协议](../docs/BENCHMARK.md)。以下路径相对于 `data/`，数据文件不随仓库分发。

## 缓存格式

```text
processed/gse240429_heg/arrays/C73_{A1,B1,C1,D1}.npz
processed/gse240429/context_20260913/C73_{A1,B1,C1,D1}_{224,896,1792}_patches.npy
```

NPZ 包含 `genes`、`barcodes` 和全分辨率坐标 `coordinates_xy`；训练和评价还需 `raw_counts`。图像数组为 `[spots,224,224,3]` 的 uint8 RGB，行顺序与 NPZ 一致。

## 源文件

表达矩阵和坐标取自 [BLEEP 数据目录](https://github.com/bowang-lab/BLEEP/tree/main/GSE240429_data)，参考提交 `2395967`。源目录结构为：

```text
filtered_expression_matrices/{1,2,3,4}/{matrix.mtx,features.tsv,barcodes.tsv}
tissue_pos_matrices/tissue_positions_list_{1,2,3,4}.csv
```

H&E 取自 [GSE240429](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE240429) 的 GSM7697868–GSM7697871，依次对应 A1–D1。将 `GEX_C73_*_Merged.tiff.gz` 解压为可内存映射的 RGB TIFF，命名为 `C73_A1.tif` 等，放入 `processed/gse240429/tiff/`。裁图坐标以全分辨率图像为准。

## 生成缓存

在仓库根目录运行：

```bash
python run.py prepare --source third_party/BLEEP/GSE240429_data/data
```

程序核对 A1+B1 的高表达基因排名，按固定列顺序写入表达数组和三尺度图像缓存。已有输出不覆盖；可用 `--arrays`、`--images` 指定新目录，并同步修改训练配置中的路径。仅重建表达数组时加 `--arrays-only`。
