# PCC 诊断与实现修正

2026-09-13 的检查涉及基因选择、归一化和推理实现。对应结果见[基线表](../benchmarks/liver/baseline_20260913.csv)。

## 基因面板

早期脚本按 log 表达方差选基因，并排除 MT、RPL、RPS 基因。所得 200 个基因与训练集平均表达最高的 200 个基因无重叠。

修正后，在 A1+B1 上按全转录组归一化后的平均 log 表达选择 HEG200，其中包括 9 个线粒体基因和 47 个核糖体基因。HEG50 为同一排序的前 50 个。D1 的零计数比例由旧面板的 24.70% 降至 3.45%。新旧面板的分数对应不同基因集合。

基因列表及训练统计见 [genes.csv](../benchmarks/liver/genes.csv)。

## 评价目标

早期结果混用了全转录组归一化、面板归一化和原始计数。当前人肝主表统一采用 HEG200 内部的 log 相对表达，定义见[基准协议](BENCHMARK.md)。

ResSAT 的论文指标使用 PCA/Harmony 重建表达。固定同一份 SP 预测，将参照从重建表达改为未重建 log 表达后，HVG2000 PCC 从 0.6667 变为 0.3240；后者有 1974 个有效基因。HEG50 在未重建参照上的 PCC 为 0.7584。这一差异说明重建表达与实测表达对应不同的评价目标。

## 图像与推理实现

| 问题 | 修正与检查 |
|---|---|
| ResSAT 示例图块为 float32、0–255，直接送入 `ToPILImage` 会错误缩放 | 加载后转为 uint8；SA/SP 复现使用修正后的图像 |
| Stem 的默认采样截断范围与表达目标不符，EMA 衰减过慢 | 改用 EMA 0.99、固定验证噪声，并在 C1 比较参数和截断方式 |
| GenAR 训练未达到原计划轮数，解码方式未完成验证 | 使用已完成 40 轮的权重，按 C1 选择末级 top-8 期望解码；权重仍依据训练损失保存 |
| 早期 MLP 被用作 ST-Net 的替代 | 将 MLP 单列，增加 DenseNet121 全参数微调的 ST-Net 适配 |
| 常量列 PCC 和 Moran's I 邻居处理不一致 | 常量列记为 NaN；邻居列表明确排除自身，并以独立计算核验 |

对四张人肝 TIFF 检查数据段边界，并抽查 128 个缓存图块，均与原图按坐标裁取的结果逐像素一致。ResSAT 示例的类型问题未出现在这些人肝缓存中。

## ResSAT 鼠脑复现

[ResSAT 论文](https://doi.org/10.1186/s13059-026-04168-x)中的 0.6577、0.6980 对应 HVG2000，HEG50 对应 0.8781、0.8999。

| 数据 | 论文 HVG2000 | 本地 HVG2000 | 论文 HEG50 | 本地 HEG50 |
|---|---:|---:|---:|---:|
| SA | 0.6577 | 0.6506 | 0.8781 | 0.8803 |
| SP | 0.6980 | 0.6667 | 0.8999 | 0.8930 |

本地采用 seed 42；SA 使用作者预处理示例，SP 根据下载的原始数据重建处理流程。论文数值为多次重复均值。两份权重重新推理后与原存档预测逐元素一致。该实验与人肝任务的组织、基因和表达目标均不同。

## 记录与复核

人肝七种方法的结果见[基线表](../benchmarks/liver/baseline_20260913.csv)。本地 `results/verified_20260913/` 保存以下记录：

| 文件 | 内容 |
|---|---|
| `benchmark_predictions.npz` | 统一单位的预测、真值、基因和坐标 |
| `benchmark.csv`、`benchmark_per_gene.csv` | 平均指标与逐基因指标 |
| `fixed_HEG200_gene_audit.csv` | 训练表达排名和检测率 |
| `fresh_checkpoint_verification.json` | ResSAT 权重与重推理核验 |
| `cached_feature_verification.json` | 缓存图像特征核验 |

准备本地数据与权重后，在仓库根目录运行：

```bash
python scripts/50_audit_protocol_and_assets.py
python scripts/50b_audit_image_alignment.py
python scripts/51_verify_ressat_checkpoints.py
python scripts/52_evaluate_verified_benchmark.py
```

视野、训练目标和计数条件的对照见[多尺度实验](06_LIVER_CONTEXT_IMPROVEMENT.md)；最终结果见[六模型集成](ENSEMBLE.md)。
