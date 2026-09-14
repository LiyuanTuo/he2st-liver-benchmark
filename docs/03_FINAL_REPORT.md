# 基于 H&E 图像生成空间转录组数据：实验报告

> **2026-09-13**：本文为历史报告。当前高表达基因七法对比、已验证的错误、真实 ST-Net 适配和 ResSAT 重新推理证据见 [05_PCC_DIAGNOSIS.md](05_PCC_DIAGNOSIS.md)。原统一 Benchmark PPT 已更新为 11 页；不要把本文旧基因面板结果与新 HEG 表混排。

> **2026-09-08 更正**：下文保留此前实验记录。审计发现旧统一表中 GenAR 使用“200 基因总量”归一化，
> 其他方法使用“全部基因总量”，不能混排。新九页 PPT 对六法统一作面板内归一化后重新评测，未重训；
> 详见[新汇报与口径说明](04_BENCHMARK_PPT_NOTES.md)。编码器/预算一致的完整比较与独立患者验证尚未完成。

更新日期：2026-09-07
状态：数据下载、预处理、模型实验、外部 benchmark 与结果核验均已完成。

## 1. 结论先行

本项目完成了三个层次的验证，而不是只在一个小数据集上报一个最好数值：

1. **统一人肝实验**：在 GSE240429 的相同切片、相同 200 基因和相同评测代码下，比较
   Image Ridge、ST-Net 风格 MLP、BLEEP、ResSAT、GenAR、Stem 及非图像基线。
2. **原论文协议复现**：在 ResSAT 的两套鼠脑数据 SA、SP 上验证作者流程。SP 从 10x
   原始 H&E 与表达矩阵重新预处理，2,000 HVG PCC 达到 **0.6667**（论文 0.6980），
   top-50 HEG PCC 达到 **0.8930**（论文 0.8999）。
3. **独立外部 benchmark**：在 HEST-IDC 四折乳腺癌任务上，本地 ResNet50 + Ridge
   得到 **0.473872**，与 HEST 官方公开值 **0.4739** 四舍五入后完全一致。

统一人肝实验中，ResSAT 在两个测试切片上的严格 PCC 分别为 0.1883 和 0.0998，
平均 **0.1441**，并且在两张测试切片上都取得最高 PCC。另一方面，同一方法在 C1 与 D1 上的
差异远大于不同随机种子之间的差异，说明这个任务当前最主要的不确定性来自**切片域偏移**，
不能只凭一张测试切片判断模型优劣。

GenAR 和 Stem 在本项目 8 GB 显存、小样本适配条件下没有复现论文优势。这里应解释为
“当前数据规模、特征和训练预算下的负结果”，不能据此否定它们在论文的大数据与基础模型
特征条件下的结论。

## 2. 任务是什么

空间转录组在一张组织切片的许多空间位置（spot）测量基因表达。监督学习数据中的每个
spot 同时具有：

- 以该 spot 为中心的 H&E 图像块；
- 它在整张切片中的坐标；
- 实验测得的基因表达向量。

模型学习以下映射：

```text
整张 H&E + spot 坐标
        ↓ 按坐标裁图
每个 spot 的 H&E patch
        ↓ 模型
预测矩阵 [spot 数 × gene 数]
        ↓ 将某一基因列画回坐标
该基因的空间表达热图
```

因此输出不是一张新的染色图，而是一个预测的 `spot × gene` 数值矩阵。它属于
in-silico prediction，不能替代 RNA 实验测量。

## 3. 数据与可复现资产

| 数据 | 组织/平台 | 本项目用途 | 实际进入实验的规模 |
|---|---|---|---:|
| GSE240429 | 人肝，10x Visium，A1–D1 | 六类方法统一比较、切片稳健性 | 9,269 spots；固定 200 genes |
| ResSAT-SA | 鼠脑矢状前部，10x Visium，2 sections | 作者处理后示例复现 | test 2,459 / train 2,251 spots；2,000 HVGs |
| ResSAT-SP | 鼠脑矢状后部，10x Visium，2 sections | 从原始数据重建作者协议 | test 2,945 / train 2,976 spots；2,000 HVGs |
| HEST-IDC | 乳腺浸润性导管癌，Xenium，4 slides | 独立官方四折 benchmark | 35,536 patches；固定 50 genes |

### 3.1 新补齐的数据

- `data/raw/ressat_mouse_brain/`：SA1、SA2、SP1、SP2 的 10x 官方 H&E TIFF、
  filtered feature matrix 和 spatial 元数据；`manifest.json` 记录文件大小与 SHA-256。
- `data/processed/ressat_original_rebuilt/`：从上述原始文件按论文 Methods 重建的
  QC、2,000 HVG、PCA-50、Harmony 和图像 patch 数据。
- `data/raw/hest_bench/IDC/`：HEST-IDC 的 4 个 patch HDF5、4 个表达 h5ad、4 折
  train/test CSV、固定基因列表与 SHA-256 清单。只下载完成任务所需的约 5.5 GB
  子集，没有下载超过 2 TB 的整个 HEST 数据集合。

### 3.2 数据审计发现

1. **GSE240429 GEO 总包截断**：总包中的 D1 TIFF 缺少尾部约 343 MB。本项目从
   GEO 样本 FTP 单独补下，并用 gzip ISIZE 与解压结果核验，四张图像现均完整。
2. **ResSAT 论文与当前 10x 文件不完全一致**：实际下载的 SA 文件为 32,285 genes，
   SP 为 31,053 genes，与论文 Methods 中的 SA/SP 数字次序相反；原始 spot 数也与文中
   个别数字不同。报告以当前公开源文件及 SHA-256 为准，不把推测写成事实。
3. **ResSAT 未公开完整预处理代码**：本项目按 Methods 重建后，SA2 保留 2,251 spots，
   与作者示例完全一致；SA1 保留 2,476，而作者示例为 2,459，相差 17。重建与作者示例
   的 HVG 重叠 1,876/2,000（93.8%）。因此 SP 实验称为“Methods-based reconstruction”，
   不冒充无法证明的逐字节复原。

## 4. 实验设计

### 4.1 GSE240429 统一协议

主协议固定 `train=A1+B1`（4,727 spots），并交换 C1/D1 的验证和测试角色：

| 实验 | 训练 | 验证 | 测试 |
|---|---|---|---|
| fold C1 | A1+B1 | D1 | C1（2,277 spots） |
| fold D1 | A1+B1 | C1 | D1（2,265 spots） |

200 个基因只用训练切片选择；图像输入为每个 spot 周围的 224×224 patch。所有统计量、
PCA 和模型参数只在训练数据拟合。按整张切片留出，避免把相邻 spot 随机分到训练和测试
造成空间泄漏。

MLP 与 BLEEP 各运行 5 个种子（11、22、33、44、55），同时报告单模型均值/标准差和
5 模型集成。Ridge 超参数、BLEEP 邻居数及神经网络早停均由验证切片选择，测试标签不参与。

### 4.2 ResSAT 作者协议

```text
H&E patch → ResNet50 图像特征 ─┐
                                ├→ FiLM 融合 → spot 自注意力 → 50 维表达表示
归一化坐标 → Fourier 位置编码 ─┘                              ↓
                                                 inverse PCA → 2,000 genes
```

- SA：直接使用作者 Zenodo 处理后示例，Section 2 训练、Section 1 测试。
- SP：从当前 10x 原始文件重建，Section 2 训练、Section 1 测试。
- 训练：ResNet50、100 epochs、batch 32、学习率 5e-4、seed 42。
- 同时计算两个口径：论文的“预测和观察值均在 PCA/Harmony 重建空间中比较”，以及
  对未经过 PCA 重建的完整 log-normalized 表达真值进行严格比较。

### 4.3 HEST-IDC 官方四折协议

四张切片每次留出一张测试，其余三张训练。固定使用官方 `var_50genes.json`；图像编码
后依次做 StandardScaler、PCA-256、无截距 Ridge，PCC 为 50 个基因逐基因 PCC 的宏平均。

对比两个编码器：

- `resnet18_shared`：本项目统一人肝实验使用的轻量 ImageNet 编码器，512 维；
- `resnet50_hest`：与 HEST/TRIDENT 公开实现一致的 ResNet50 layer3 + global average
  pooling，1,024 维。

## 5. 统一人肝实验结果

### 5.1 固定测试切片 D1 的完整方法比较

下表全部使用相同 D1 真值和固定 200 基因。`train_gene_mean` 是常数预测，因此 PCC 无定义。

| 方法 | PCC ↑ | Spearman ↑ | MAE ↓ | RVD ↓ | Moran corr ↑ |
|---|---:|---:|---:|---:|---:|
| train gene mean | — | — | **0.582** | 1.000 | — |
| coordinate Ridge | −0.005 | 0.005 | 0.589 | 0.950 | 0.039 |
| coordinate KNN | −0.002 | 0.005 | 0.595 | 0.899 | 0.279 |
| Image Ridge | 0.059 | 0.047 | 0.607 | 0.833 | 0.309 |
| ST-Net 风格 MLP | 0.057 | 0.044 | 0.609 | 0.803 | **0.446** |
| BLEEP | 0.052 | 0.048 | 0.626 | 0.800 | 0.175 |
| **ResSAT** | **0.100** | **0.073** | **0.579** | 0.914 | −0.008 |
| GenAR | 0.005 | −0.004 | 1.341 | 0.883 | −0.066 |
| Stem | 0.003 | 0.003 | 1.044 | **0.401** | 0.050 |

不能只凭某一列判定方法“全面最好”：ResSAT 的逐基因相关和 MAE 最好，但 MLP 的
Moran's I 保持最好；Stem 的 RVD 看起来较低却几乎没有逐基因预测信号，证明方差大小
接近不等于空间位置预测正确。

### 5.2 两张测试切片与随机种子稳健性

| 方法 | test C1 PCC | test D1 PCC | 两切片平均 |
|---|---:|---:|---:|
| Image Ridge | 0.1046 | 0.0589 | 0.0818 |
| MLP，5-seed ensemble | 0.0964 | 0.0712 | 0.0838 |
| BLEEP，5-seed ensemble | 0.1186 | 0.0576 | 0.0881 |
| **ResSAT** | **0.1883** | **0.0998** | **0.1441** |

单模型的种子波动较小：MLP 在 C1/D1 上分别为 0.0843±0.0029、0.0618±0.0046；
BLEEP 分别为 0.1055±0.0041、0.0516±0.0072。相比之下，同一方法换一张测试切片可变化
0.04–0.09。主要结论是：**切片域偏移大于随机初始化误差**，未来应扩大患者和切片数量，
以患者级/切片级交叉验证给出置信区间。

![两张测试切片上的稳健性](../figures/06_cross_slide_robustness.png)

### 5.3 配对逐基因比较：优势是否由少数基因偶然造成

为避免只比较四个平均数，本项目把 200 个固定基因在 C1、D1 上的 PCC 组成
`200 genes × 2 slides = 400` 个配对单元，并进行 20,000 次配对 Bootstrap。ResSAT
相对 Image Ridge、MLP ensemble、BLEEP ensemble 的平均 PCC 差分别为
**+0.0623 [0.0579, 0.0668]**、**+0.0602 [0.0550, 0.0654]**、
**+0.0560 [0.0520, 0.0600]**；ResSAT 在 92.8%、85.5%、91.8% 的配对单元上更高。

![逐基因配对比较](../figures/09_gene_level_statistical_comparison.png)

这里的区间只能回答“当前两张切片、固定 200 基因上的差异是否稳定”，属于**基因层面的
描述性区间**。因为只有一位患者的两张测试切片，基因并非独立患者重复，不能把该区间解释为
患者总体的置信区间，也不能写成临床统计显著性。

### 5.4 为什么生成模型没有赢

- GenAR 学到了基因总体高低顺序，但计数分布塌缩：预测零值 3.7%，真值 24.7%；
  预测最大计数 19，真值 152，导致跨 spot 的逐基因 PCC 接近 0。
- Stem 的验证去噪损失下降，但从噪声反向采样后 PCC 接近 0。小样本下模型可主要学习
  高噪声时复制输入噪声，而没有真正学习 H&E 条件到表达的映射。
- 本项目只有 4,727 个训练 spot，使用 ResNet18 特征与 8 GB 单卡适配；GenAR/Stem
  论文使用病理基础模型特征、更长训练及多卡高端 GPU。因此这些结果回答的是“本项目
  条件下是否可用”，并非等价复现论文的大规模设置。

## 6. ResSAT 原论文数据复现

| 数据/指标 | 论文 | 本项目 | 绝对差 |
|---|---:|---:|---:|
| SA，2,000 HVGs PCC | 0.6577 | 0.6506 | −0.0071 |
| SA，top-50 HEGs PCC | 0.8781 | 0.8803 | +0.0022 |
| SP，2,000 HVGs PCC | 0.6980 | 0.6667 | −0.0313 |
| SP，top-50 HEGs PCC | 0.8999 | 0.8930 | −0.0069 |

![ResSAT 原论文数据复现](../figures/07_ressat_original_protocol.png)

这组结果支持“ResSAT 流程可以在公开数据上复现并接近论文结论”。SP 的全基因差距稍大，
与作者未提供精确预处理代码、当前源文件版本不同及单种子有关；top-50 结果仍非常接近。

更重要的是，SP 在论文 PCA/Harmony 重建空间中的 PCC 为 0.6667，而对未经过 PCA 重建的
完整 log-normalized 真值为 **0.3240**。这不是模型突然变差，而是评价对象不同：PCA-50
只保留主要变异并同时平滑预测和观察值，会使相关性明显升高。因此报告论文数字时必须写清
“reconstructed batch-corrected space”，不能与完整表达矩阵 PCC 混为一谈。

另一个实现审计是：官方 ResSAT 的自注意力沿 DataLoader batch 维运算，而不是显式空间
邻域。训练又使用 shuffle，因此批内 spot 不一定相邻。本项目把推理 batch 从 1 改到 64，
PCC 几乎不变，说明当前权重的注意力近似弱交互；设计上仍建议改为固定 kNN/半径邻域图，
获得确定且可解释的空间上下文。

## 7. HEST-IDC 外部 benchmark

| 编码器 | fold 0 TENX99 | fold 1 TENX95 | fold 2 NCBI785 | fold 3 NCBI783 | 平均±标准差 |
|---|---:|---:|---:|---:|---:|
| ResNet18，本项目轻量编码器 | 0.3642 | 0.4678 | 0.4355 | 0.4069 | 0.4186±0.0381 |
| **ResNet50，HEST 协议** | 0.4081 | 0.5138 | 0.4555 | 0.5181 | **0.4739±0.0453** |

HEST 当前公开的 ResNet50 IDC 参考值为 0.4739；本地精确值 0.473872。二者一致，说明
本项目 HEST 数据、基因、四折拆分、特征层和 Ridge 评测均正确对齐。ResNet50 比共享
ResNet18 高 0.0552，也表明编码器选择本身是重要实验变量。

![HEST-IDC 外部 benchmark](../figures/08_hest_idc_external_benchmark.png)

这项实验只比较**冻结图像编码器 + 官方线性探针**，不等同于把 ResSAT、GenAR、Stem
全部迁移到 HEST。它的作用是提供独立数据上的协议核验和编码器基线。

## 8. 工程优化与可复现性

### 8.1 为什么需要缓存

原始 TIFF/HDF5 解析和 CNN 特征提取很耗时，而同一图像 patch 会被多个方法和多个种子
反复使用。本项目将下列确定性中间结果缓存：

```text
原始 H&E + spot 坐标
        ↓ 只做一次
对齐后的 patch HDF5 / dataset.pkl
        ↓ 只做一次 CNN 前向
ResNet18 或 ResNet50 feature cache
        ↓ 重复利用
Ridge / MLP / BLEEP / 四折评测
```

缓存不是修改模型算法，而是避免每个 epoch、每个种子和每种下游方法重复读取超大图像并
运行相同 CNN。缓存文件同时保存 barcode，加载时再与 h5ad 索引对齐，防止表达标签错位。

### 8.2 消费级 GPU 适配

- Stem 在 fp32 大 batch 下峰值显存超过 8 GB并出现严重爬行；改为 AMP(fp16)+batch 32
  后峰值约 2 GB，单步由秒级降到约 0.07 秒。
- GenAR 使用 mixed precision 和较小 batch；训练预算由论文/仓库大规模配置降为 40 epochs。
- ResSAT-SP 的 100 epoch 完整训练耗时约 25.1 分钟。
- HEST 35,536 个 patch 的两套编码器特征提取与四折回归总计约 207 秒；再次运行会直接
  读取 feature cache。

## 9. 一键复查关键新增实验

以下命令假定数据已经下载，不包含联网下载步骤：

```powershell
cd "D:\Code\H&E Generation"

# 人肝两测试切片 + 5 随机种子
python scripts\19_cross_slide_robustness.py

# WSL CUDA：ResSAT 第二个人肝 fold
wsl bash -lc "cd '/mnt/d/Code/H&E Generation' && python3 scripts/20_train_ressat_second_fold.py"

# WSL CUDA：从 ResSAT 原始 10x 数据重建预处理并训练 SP
wsl bash -lc "cd '/mnt/d/Code/H&E Generation' && python3 scripts/21_prepare_ressat_original_datasets.py && python3 scripts/22_train_ressat_sp_rebuilt.py"

# WSL CUDA：HEST-IDC 两套编码器 + 官方四折线性探针
wsl bash -lc "cd '/mnt/d/Code/H&E Generation' && python3 scripts/24_run_hest_idc_benchmark.py"

# 汇总绘图
python scripts\25_make_extended_results.py

# 两张测试切片上的逐基因配对 Bootstrap
python scripts\26_statistical_comparison.py
```

已有缓存和 checkpoint 时，部分脚本会复用结果；精确配置保存在各自的 JSON、Lightning
日志和 checkpoint 目录中。

## 10. 最终判断与下一步

### 已经可以得出的结论

1. H&E 确实含有可预测的基因表达信号，但跨切片泛化明显比单切片内拟合困难。
2. 在当前统一人肝小数据设置中，ResSAT 最有效；简单 Ridge/MLP/BLEEP 仍是很强且廉价的
   基线，复杂生成模型没有自动带来更好点预测。
3. ResSAT 在原论文两套鼠脑数据上已接近论文 PCC，且 HEST-IDC 官方数值被精确复现，
   因而当前工程流程不只是“能运行”，还具有两个独立的数值校验点。
4. 论文数值高度依赖基因集合、测试切片、特征编码器和表达评测空间。脱离这些条件直接比较
   PCC，会得到误导性结论。

### 后续最值得投入的方向

1. 增加患者级数据并做 nested leave-one-patient-out，给出切片间置信区间。
2. 将 ResSAT 的 batch attention 改为固定空间 kNN/半径图，并做 image-only、position-only、
   image+position 的消融。
3. 将共享 ResNet18 替换为可公开获得的病理基础模型，再判断 GenAR/Stem 的负结果中有多少
   来自条件特征不足。
4. 除 PCC 外加入细胞类型解卷积、通路富集和空间域识别，检验预测是否支持实际生物学分析。
5. 对生成模型报告多次采样的校准、覆盖率和不确定性，而不是把随机样本均值当成普通回归结果。

## 11. 证据入口

- 结果总览：[results/RESULTS_REPORT.md](../results/RESULTS_REPORT.md)
- 关键数值汇总 JSON：[results/extended_summary.json](../results/extended_summary.json)
- 原统一比较：[results/unified_comparison.md](../results/unified_comparison.md)
- 跨切片与多种子：[results/cross_slide_robustness/metrics.json](../results/cross_slide_robustness/metrics.json)
- 配对逐基因比较：[results/statistical_comparison.json](../results/statistical_comparison.json)
- ResSAT-SP：[results/ressat_sp_rebuilt/evaluation.json](../results/ressat_sp_rebuilt/evaluation.json)
- HEST-IDC：[results/hest_idc/evaluation.json](../results/hest_idc/evaluation.json)
- 数据下载校验：[data/raw/hest_bench/IDC/manifest.json](../data/raw/hest_bench/IDC/manifest.json)
- 论文与官方仓库版本：[papers/README.md](../papers/README.md)

主要参考：ResSAT 正式版（Genome Biology, 2026）、BLEEP（NeurIPS, 2023）、
Stem（ICLR, 2025）、GenAR（Medical Image Analysis, 2026）、HEST-1k（NeurIPS, 2024）
及 ST-Net（Nature Biomedical Engineering, 2020）。本地 PDF 和官方代码 commit 均已固定在
`papers/` 与 `third_party/`。
