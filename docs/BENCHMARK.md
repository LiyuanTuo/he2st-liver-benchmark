# 人肝基准

GSE240429 中供体 C73 的四张连续人肝切片用于跨切片预测。H&E 来自 [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE240429)，表达矩阵与坐标来自 [BLEEP](https://github.com/bowang-lab/BLEEP/tree/main/GSE240429_data)。spot 指空间转录组的测量位置。

| 切片 | spot 数 | 用途 |
|---|---:|---|
| C73_A1 | 2378 | 训练 |
| C73_B1 | 2349 | 训练 |
| C73_C1 | 2277 | 验证及模型选择 |
| C73_D1 | 2265 | 测试 |

## 基因选择与表达单位

在 A1+B1 上，将每个 spot 的全转录组计数归一化到 10000，取 `log1p` 后按训练平均表达降序选择 HEG200；前 50 个为 HEG50。空基因名剔除，重复名保留首次出现，核糖体和线粒体基因参与排序。

[genes.csv](../benchmarks/liver/genes.csv)保存基因列表和 HEG50 标记，列顺序沿用 GenAR 的层次排序。列表校验值保存在[集成配置](../benchmarks/liver/final_ensemble.json)的 `genes_sha256` 字段中。

预测目标为 HEG200 内部的相对表达：

```text
y[i,g] = log(1 + 10000 * count[i,g] / sum(count[i, HEG200]))
```

选基因以全转录组总计数为分母，评价以 HEG200 总计数为分母。测试表达不做 PCA、Harmony 或邻域平滑；HEG50 直接取相应列，不再归一化。ContextFusion 直接预测上述表达值，其他方法的转换见[方法适配](02_METHODS_AND_PROTOCOL.md)。

## 指标与模型选择

PCC（Pearson 相关系数）按基因在全部测试 spots 上计算，再对 HEG200、HEG50 分别取平均。常量向量记为 NaN，另报有效基因数。MAE 为同一表达单位下的平均绝对误差。

ContextFusion 的模型权重、旋转增强和集成比例按 C1 HEG200 PCC 选择；其他方法见[适配配置](02_METHODS_AND_PROTOCOL.md)。D1 在开发中多次使用，尚无独立供体验证。

各方法共用数据划分、基因集合和评价目标，但编码器、视野及训练预算不同。ResSAT 鼠脑重建表达实验单独记录于 [PCC 诊断](05_PCC_DIAGNOSIS.md)。
