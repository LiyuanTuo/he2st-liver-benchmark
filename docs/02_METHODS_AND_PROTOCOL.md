# 方法、模型与统一实验协议

## 1. 为什么不能直接抄论文表格比较

各论文看似都报告 PCC，但实验对象并不完全相同：

- ResSAT 正式版预测 50 维 Harmony-corrected PCA 表示，再重建 2,000 个 log-normalized HVG；
- Stem 生成 log 表达分布，并提出相对方差距离 RVD；
- GenAR 直接生成 200 个基因的原始整数计数，再在 log 空间评测；
- BLEEP 用训练库中的近邻表达插补，不直接拟合逐基因回归头；
- 图像编码器可能是 ResNet、UNI 或 CONCH，基因数、组织和切片划分也不同。

因此本项目提供两类结果：

1. **作者协议验证**：在作者示例数据上运行作者代码，回答“官方流程能否跑通”；
2. **统一协议比较**：所有方法使用相同 GSE240429 切片、200 个训练集选择的基因、相同图像特征和同一测试切片，回答“在同一条件下谁更好”。

两类数值不会放进同一排行榜。

## 2. 经典回归基线：ST-Net 风格

```text
224×224 H&E patch → CNN/病理编码器 → 图像向量 → MLP → G 个连续表达值
```

优点是简单、训练快、输出稳定，适合判断复杂生成模型是否真的带来收益。局限是每个 spot 独立处理，不建模邻域和基因之间的联合分布。

本项目同时报告三条“不看图像”的下限：训练集基因均值、坐标 Ridge、坐标 KNN。复杂模型必须明显超过它们，才能说明 H&E 形态提供了信息。

## 3. BLEEP：图像—表达对比学习与近邻插补

```text
训练：H&E encoder ─┐
                   ├─ 对比损失 → 共享嵌入空间
      expression ──┘

推理：测试 H&E → 图像嵌入 → 检索 k 个训练 spot → 加权平均其表达
```

优势：无需为每个基因设置独立输出头，结果可追溯到参考 spot。局限：检索成本和参考库规模相关；输出是训练表达的组合，可能过度平滑；跨组织/平台时共享嵌入不一定对齐。

## 4. ResSAT：形态、坐标和 spot 交互

正式版流程是：

```text
H&E patch → fine-tuned ResNet50 → 512-d image feature f
                                              │
normalized (x,y) → random Fourier features → 128-d spatial feature s
                                              │
                    FiLM: f' = f ⊙ γ(s) + β(s)
                                              │
                             concat [f', s] → 640-d
                                              │
                          self-attention across spots
                                              │
                  MLP → 50-d Harmony/PCA expression target
                                              │
                        inverse PCA → 2,000-gene profile
```

论文正式版的数据处理为：每 spot 裁 H&E；坐标按切片 min-max 到 `[0,1]`；计数归一到 10,000 后 log；批次感知选择 2,000 HVG；逐基因缩放；PCA 到 50 维；Harmony 校正后作为标签。损失为 50 维目标上的 MSE。

### 官方实现需要特别审计的地方

**审计发现 A（数据格式陷阱，已验证）：官方示例 patch 格式与新版 torchvision 不兼容。**

官方 Zenodo 示例的 `dataset.pkl` 存的是 float32 HWC、值域 0–255 的 patch。
新版 torchvision 的 `ToPILImage` 对浮点数组按 `[0,1]` 处理（先 ×255 再转 uint8），
于是 0–255 的 float patch 会先 ×255 再按 uint8 回绕（例如 200 → 56），
图像被**静默损坏**——训练/推理仍能跑完，指标看起来也"正常"（鼠脑坐标本身高度预测表达），
但图像信息已被破坏。本项目实测：把官方 patch 直接过官方 transform，输出是回绕噪声而非原图。

修复方式（不改第三方代码）：加载后把 patch 就地转成 `uint8`（uint8 原样通过 ToPILImage，
任何 torchvision 版本行为一致）。本项目官方示例的复现训练（`official_seed42` version_1）
和统一数据集（`scripts/11`）都使用该修复。修复前的 version_0 权重保留在
`results/ressat_official/` 中作为审计对照，不进入结论。

**审计发现 B（官方实现设计问题）：`SA_Feature` 的注意力作用在 DataLoader batch 上。**

官方 `SA_Feature` 接收形状 `[batch, feature]`，其 `QKᵀ` 是 `[batch,batch]`，所以交互对象取决于当前 DataLoader 批次，而不是固定空间邻域：

- 训练 loader 使用 `shuffle=True`，同一 batch 可混入不同切片的随机 spot；
- `section_id` 和 `spot_id` 虽被读入，但没有用于 attention mask；
- 推理结果理论上会随 batch size、样本顺序和同批其他 spot 改变。

**实测结果：在官方示例和本项目统一数据上，推理 batch size 1/8/16/32/64 的 PCC 几乎不变
（官方示例约 0.6512–0.6513，统一数据见 `results/ressat_unified/evaluation.json`）。**
说明该模型学到的注意力近似恒等/弱依赖，batch 敏感性在实践中影响很小，但该设计仍缺乏
可解释的空间邻域语义；改进版应按切片/空间邻域建立 attention mask，或使用 kNN/半径图，
使每个 spot 的上下文确定且可解释。

另一个风险是论文描述先合并切片再做 HVG、PCA 和 Harmony。若外层测试切片参与这些步骤，就使用了测试表达分布。统一协议中所有基因选择和归一化统计只在训练切片拟合。

### 本项目统一协议适配（scripts/11、scripts/12）

- 数据（`data/processed/ressat_gse240429/`）沿用官方格式：`Section_i/dataset.pkl`
  （patch + 50 维 PCA 目标）、`locations.pkl`（[0,1] 归一化坐标）、`pca_info.pkl`、
  `gene_list.csv`；Section 映射为 Section_1=D1(测试)、Section_2=C1(验证)、
  Section_3/4=A1/B1(训练)；
- 差异：统一 200 基因面板（官方 2,000 HVG + Harmony），PCA-50 只用训练切片拟合
  （200 基因上 50 维解释了 38.4% 方差，说明统一面板下的 PCA 瓶颈比官方更紧）；
- patch 用 scripts/05 的 224×224 缓存，存为 uint8（规避审计发现 A 的回绕问题）；
- 训练：100 epochs、lr 5e-4、batch 16（8 GB 适配，官方 batch 32）、Fourier 128、
  sigma 1、dropout 0.3、patience 10，其余保持官方代码；评测对推理 batch
  1/8/16/32/64 做稳定性审计，主口径取 batch 16。

## 5. Stem：条件扩散生成

Stem 不直接输出一个确定向量，而是学习给定 H&E 特征时表达分布的 score/noise：

```text
真实表达 y → 逐步加噪得到 y_t
H&E patch → UNI/CONCH embedding ─┐
time step t ─────────────────────┼→ conditional DiT/Transformer → 预测噪声
y_t ─────────────────────────────┘

推理：从噪声开始，多步去噪 → 一个表达样本；可重复采样得到不确定性
```

优势：能表达“一种形态对应多种可能分子状态”的不确定性，并关注方差/异质性而非只有均值。缺点：训练和多步采样昂贵；结果依赖采样次数；若最后对大量样本取均值，仍可能把异质性平滑掉。

Stem 的 RVD 衡量逐基因预测方差与真值方差的相对偏差。本项目保留 RVD，并同时报告空间 Moran's I，避免一个方差数值掩盖空间位置错误。

### 本项目统一协议适配（scripts/13、scripts/14）

官方 `stem_train.py` 硬编码 UNI+CONCH 拼接特征目录、图像增强倍数与 `cuda:6`/DDP，
不能直接用于单卡消费级 GPU。本项目**复用官方模型与扩散代码**（`Stem/models.py`、
`Stem/diffusion/`），替换数据管线：

- 条件：本项目统一 ResNet18-512 特征（官方 UNI+CONCH），用训练集统计量 z-score；
- 目标：统一 log1p(1e4 归一化) 200 基因面板（官方 log2(count+1) 原始计数）；
- DiT 12 层 × 384 隐层 × 6 头（仓库默认），**AMP(fp16) + batch 32 + 300 epochs**
  （8 GB 适配；实测 fp32 前向图在 batch>32 时显存溢出并显著变慢——前向图约
  6 GB/批 @128，完整损失路径峰值 12 GB 超过 8 GB 物理显存），AdamW 1e-4，
  EMA 0.9999，每 5 个 epoch 用验证切片损失做早停（patience 100）；
- 采样：**100 步 DDIM（eta=0）+ clip_denoised=True**（8 GB 适配实测：欠训练模型的
  高噪声步 x0 估计会爆炸——不裁剪时样本范围 ±900，裁剪后落在 [-1,1]），每个测试
  spot 采样 **3** 次；点预测取 3 个样本均值，不确定性用逐基因样本标准差与真值
  标准差之比（RVD(samples)）报告。

**适配结果审计（重要）**：统一数据上训练 130 epochs 早停后（val 损失 0.153），
模型采样出的样本**没有任何预测信号**（各种采样配置下宏平均 PCC≈0）。原因是
扩散损失的 0.15 主要由高噪声步主导——高噪声步下 `x_t ≈ 噪声`，"复制输入"策略
即可拿到低损失，模型未被迫学习条件映射；33M 参数 DiT 在 4,727 个训练 spot 上
以 300-epoch 级预算（论文为 4,000 epochs + 8× 增强 + A100）远不足以学会
图像→表达的条件分布。这是"生成式模型复杂度与数据规模不匹配"的直接证据，
在最终报告中如实报告为负结果，不以更换采样配置掩盖。

## 6. GenAR：层次基因组上的粗到细离散生成

GenAR 针对两个问题：独立回归忽略基因共表达；连续 log 表达与测序的离散计数不一致。

```text
H&E foundation-model feature + normalized coordinate → condition embedding

训练切片的 200 基因按共表达层次排序
1 gene-group summary
  → 4 groups
    → 8 groups
      → 40 groups
        → 100 groups
          → 200 genes (最终原始整数 count token)
```

每一层把较粗预测作为下一层条件，词表默认是整数 `0..2000`。优点是显式建模跨基因依赖、保留零计数和物理计数尺度；缺点是输出词表大、Transformer 计算和显存要求高，计数上限会截断高表达值，且层次聚类依赖训练数据。

### 论文与仓库配置不一致

当前论文 PDF 附录写的是 768 维、batch 256、50 epochs；公开仓库的“paper configuration”写的是 512 维、全局 batch 64、200 epochs。主文又写 batch 64。项目会：

- 保存实际运行时完整配置；
- 把“仓库默认复现”和“8 GB 适配配置”分开命名；
- 不把任一配置未经说明地称为唯一论文配置。

### 本项目统一协议适配（scripts/09、scripts/10）

- 通过运行时向 `configs.DATASETS` 注入 `gse240429`（dir_name、val=C73_C1、
  test=C73_D1、recommended_encoder=resnet18），不改动第三方代码；
- 数据目录 `data/processed/gse240429/` 直接符合官方布局：`st/*.h5ad`（原始整数
  计数 + `obsm['spatial']`）、`processed_data/{all_slide_lst,selected_gene_list}.txt`
  与 `spot_features_resnet18/{slide}_resnet18.pt`（scripts/05 生成，与 h5ad 行序一致）；
- 官方 `STDataset` 不依赖 `clustering_info.json`（仅 preflight 校验用），基因顺序
  由已记录的 `selected_gene_list.txt`（SHA256 `96aab61d…c57b9c7`）保证；
- 8 GB 适配：全局 batch 16（论文 64）、40 epochs（论文 200）、precision 16-mixed，
  其余超参保持仓库 paper 配置（512/8/8、scales (1,4,8,40,100,200)、count cap 2000、
  Adam 1e-4、seed 2021）。该配置只是“统一数据上的适配实现”，不冒充论文数值。

## 7. 统一评测指标

### 主指标

- `PCC macro, fixed genes`：预先固定的 200 基因逐基因 PCC 的宏平均；
- `Spearman macro, fixed genes`；
- log-normalized `MAE / RMSE`；
- `RVD`：预测与真值逐基因方差的相对距离；
- Moran's I 的误差与跨基因相关：空间结构是否保留。

### 生成计数的附加指标

- raw-count MAE/RMSE；
- 真值与预测零值比例；
- 非整数值比例、负计数比例；
- library depth 与均值—方差关系。

### 论文口径但不作为主结论

GenAR/Stem 的 `PCC-10/50/200` 是先在测试结果上按 PCC 排序再取最好 k 个基因。它适合复核论文表格，但 `PCC-10` 会随方法选择不同基因，天然偏乐观。本项目把它命名为 `paper_pcc_top*_test_selected`，防止误读成预先固定的 top-k 面板。

## 8. 未按官方代码运行的方法及原因（范围决策）

- **ST-Net 官方代码**：绑定作者自己的 Mendeley 乳腺癌数据集与 Unix 工具链
  （`env`、`bin/create_tifs.sh` 的 jpeg→tif 转换、患者级交叉验证），预处理不可迁移；
  本项目以 scripts/07 的 CNN 特征 + 回归头基线（ST-Net 风格）代表该类方法。
- **BLEEP 官方训练入口**：`BLEEP_main.py` 硬编码 SLURM 环境变量与 DDP，把
  A1/B1/D1 三张切片的 spot 混在一起做 80/20 **随机拆分**（同一张切片的 spot 会
  同时进入训练和测试，属于统一协议明确禁止的泄漏），目标为 3,467 维 harmony
  嵌入且需要 cv2 直接读 2.5 GB 全图；本项目以 scripts/07 复用官方
  `ProjectionHead` 与软标签对比损失的统一协议版本代表 BLEEP，并在报告中标明差异。
  若后续需要"作者协议"数值，可作为独立实验补齐。

## 9. 当前统一数据协议

| 集合 | 切片 | spot 数 | 用途 |
|---|---|---:|---|
| 训练 | C73_A1 + C73_B1 | 4,727 | 选基因、拟合预处理和模型 |
| 验证 | C73_C1 | 2,277 | 选超参数、早停 |
| 测试 | C73_D1 | 2,265 | 最后一次无偏评测 |

200 个基因从训练切片中按 log-normalized 方差选择，要求至少在 5% 训练 spot 检出，排除重复 symbol、线粒体和 RPS/RPL 核糖体基因，再调用 GenAR 官方两阶段 KMeans 形成基因顺序。最终顺序 SHA256：`96aab61d34f377a22f03fa1d3b6b70c8025cf7dbb4c4fcf9f03df4158c57b9c7`。

上述 D1 fold 保留为所有方法的完整统一排行榜。为检查结论是否依赖单张测试切片，
新增反向 fold：训练仍为 A1+B1，但 validation=D1、test=C1。Image Ridge、MLP、
BLEEP 和 ResSAT 均在两个测试切片上报告；MLP/BLEEP 另运行 5 个随机种子。完整结果见
`results/cross_slide_robustness/metrics.json` 与 `results/ressat_second_fold/evaluation.json`。

## 10. 论文口径参考数值（用于"与论文数值对照"，不直接比较）

各论文的实验对象不同，以下数值只用于理解"论文在什么数据、什么口径下达到什么水平"：

- **ResSAT（2024 预印本 / 2026 正式版）**：10x Visium 鼠脑 SA/SP 切片，
  leave-one-section-out（2 切片互训互测），报告全基因平均 PCC 与 top-50 高表达
  基因 PCC（数值在 PDF 表 1/2）。本项目在 SA 作者处理后示例和 SP 原始数据重建上
  均完成该口径：SA 0.6506（论文 0.6577），SP 0.6667（论文 0.6980）。
- **GenAR（MedIA 2026）**：UNI 特征 + 原始计数 token，PCC-10/50/200 为**在测试
  结果上按 PCC 排序后取 top-k**（偏乐观口径）。论文报告：PRAD
  0.842/0.784/0.663；HER2ST 0.702/0.650/0.512；kidney 0.589/0.514/0.354；
  mouse brain 0.568/…；ccRCC 0.457/0.394/0.276。
- **Stem（ICLR 2025）**：Kidney Visium 与 HER2ST，log 变换空间 PCC-k + MAE/MSE +
  自定义 RVD；论文强调 PCC 会漏报"过度平滑"预测，必须看 RVD。
- **BLEEP（NeurIPS 2023）**：主实验即 GSE240429 人肝（本项目同源数据），
  但其官方训练入口做跨切片随机 80/20 拆分（泄漏），且目标为 3,467 维 harmony
  嵌入——与统一协议不可直接比。
- **ST-Net（Nat. BME 2020）**：作者自己的乳腺癌数据，患者级交叉验证。

结论性对比只能在本项目统一协议内部做；与论文数值的差异要归因到基因面板、
特征编码器、切片划分与组织差异，而不是简单说"谁更高"。

## 11. 外部 HEST-IDC 协议核验

为避免所有结论只来自 GSE240429，本项目下载 HEST-IDC 四切片任务，完整复用其固定
50 基因、四折整切片拆分、StandardScaler、PCA-256 和无截距 Ridge。与 TRIDENT/HEST
一致的 ResNet50 layer3 1,024 维特征得到四折平均 PCC 0.473872，与公开参考 0.4739
一致；共享 ResNet18 特征为 0.418627。该实验验证数据和评测协议，同时量化了编码器差异，
但不等同于将所有生成模型迁移到 HEST。
