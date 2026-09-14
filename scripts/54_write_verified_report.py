"""Create the final Chinese diagnosis from audited numbers, not copied slide text."""
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/verified_20260913'


def read(file):
    return json.loads((ROOT / file).read_text(encoding='utf8'))


def main():
    b = read('results/verified_20260913/benchmark.json')
    p = read('results/verified_20260913/panel_audit.json')
    r = read('results/verified_20260913/ressat_recomputed.json')
    fresh = read('results/verified_20260913/fresh_checkpoint_verification.json')
    image = read('results/verified_20260913/tiff_patch_audit.json')
    genar = read('results/improvement_heg/genar/selection.json')
    stem = read('results/improvement_heg/stem/selection.json')
    stem_train = read('results/stem_adapted_heg/training.json')
    stem_diag = read('results/improvement_heg/stem/diagnostics.json')
    stnet = read('results/improvement_heg/stnet/selection.json')
    rows = []
    for name, m in b['methods'].items():
        rows.append(f'| {name} | {m["HEG200_PCC"]:.4f} | {m["HEG50_PCC"]:.4f} | {m["HEG200_Spearman"]:.4f} | {m["MAE"]:.3f} | {m["defined_HEG200"]}/200 |')
    table = '\n'.join(rows)
    official = '\n'.join(f'| {s} | {m["paper_HVG2000"]:.4f} | {m["reproduced_HVG2000"]:.4f} | {m["paper_HEG50"]:.4f} | {m["reproduced_HEG50"]:.4f} |' for s, m in r.items())
    native = '\n'.join(f'| {name} | {m["whole_transcriptome_scale_PCC_HEG200_diagnostic"]:.4f} | {m["whole_transcriptome_scale_PCC_HEG50_diagnostic"]:.4f} |' for name, m in b['methods'].items() if 'whole_transcriptome_scale_PCC_HEG200_diagnostic' in m)
    pca = '\n'.join(f'| {name} | {m["PCA_diagnostic_HEG200"]:.4f} | {m["PCA_diagnostic_HEG50"]:.4f} |' for name, m in b['methods'].items())
    ablations = '\n'.join(f'| {name} | {m["HEG200_PCC"]:.4f} | {m["HEG50_PCC"]:.4f} |' for name, m in b['same_target_ablation'].items())
    report = f'''# PCC 低的原因、修正与高表达基因实验核验

更新：2026-09-13。本报告与 `results/verified_20260913/benchmark.json` 为当前结果入口。旧报告保留历史数据，不能混入当前表格。

## 1. 先回答最关心的问题

**ResSAT 的高表达基因预测已经达到论文量级，但这不代表当前人肝任务也达到了该水平。**
本轮从既有 SA/SP checkpoint 重新推理，结果与存档预测一致；不是只引用旧 README。
用户提到的 **0.6577 / 0.6980 是 2,000 HVG 指标**；原论文 50 HEG 指标为 **0.8781 / 0.8999**。
这些数字均比较 PCA/Harmony 重建后的观察表达与预测表达。[论文 Tables 1–2 / Methods](https://link.springer.com/content/pdf/10.1186/s13059-026-04168-x_reference.pdf)。

| 数据 | 论文 HVG2000 | 本地 HVG2000 | 论文 HEG50 | 本地 HEG50 |
|---|---:|---:|---:|---:|
{official}

本地为 seed 42 单次；论文是多次重复的均值。SA 使用作者预处理示例；SP 从当前 10x 文件按 Methods 重建，不能声称逐字节预处理复现。
两套 checkpoint 新推理与旧存档的最大绝对差分别为 SA={fresh['SA']['maximum_absolute_difference_to_archive']:.8g}、SP={fresh['SP']['maximum_absolute_difference_to_archive']:.8g}。

## 2. 哪些地方确实错了

### 2.1 旧基因面板没有满足“高表达”的要求

旧 `scripts/02_prepare_expression.py` 按 log-normalized 方差排序，且排除了 MT/RPL/RPS；**不是随机选基因**，也不是全转录组高表达面板。
从原始 Matrix Market 文件重新计算，旧 200 基因与训练集最高表达的 200 基因重叠 **{p['old_panel_overlap_with_HEG200']}**，与最高表达 50 基因重叠 **{p['old_panel_overlap_with_full_top50']}**。
因此，不能在旧 200 基因里再挑 50 个就称为“全转录组最高表达的 50 个”。

当前用 A1+B1 的 `mean(log1p(10000 × counts / 全转录组总 counts))` 降序选 HEG200；同一个排序的前 50 为 HEG50。
只排除空名/重复名，不使用 C1/D1 表达选基因。基因文件 `fixed_HEG200_gene_audit.csv` 保存每个基因的训练均值、检测率和 HEG50 标记。
列表 SHA256：`{p['fixed_genes_sha256']}`。包含 9 个线粒体与 47 个核糖体基因，这符合本轮“按表达高低”定义；不能事后按结果增删。

D1 零值比例为 **24.70% → 3.45%**。旧 PPT 的 1.4% 混用了其他切片的统计，已更正。
这是面板稀疏性下降的直接证据，但仅凭这一统计不能把全部 PCC 差距归因于测量噪声。

### 2.2 把不同计分目标当成同一项指标

旧表混用了全转录组归一化、面板归一化和原计数；后来的“0.087→0.236”又同时更换基因集、重新训练、对真值做 PCA、只报告 50 个基因。
**这条箭头不能证明同一预测任务的模型 PCC 从 0.087 提高到 0.236。**

最直接的核验是保持 SP 预测文件完全相同：

- PCA/Harmony 重建真值上 HVG2000 PCC 为 **{r['SP']['reproduced_HVG2000']:.4f}**。
- 改成未重建 log-normalized 真值为 **{r['SP']['strict_truth_HVG2000_defined_macro']:.4f}**；有效基因 {r['SP']['strict_truth_defined_genes']}/2000，常数真值列 PCC 无定义。
- 同一组论文 HEG50 改用未重建真值仍有 **{r['SP']['strict_truth_same_HEG50']:.4f}**，说明不能笼统说论文高分“全是去噪造成”。

联合 HVG/PCA/Harmony 预处理使用各 section 的表达，属于转导式评测设置；新的人肝主基准仅在训练切片拟合基因选择/PCA。两者不代表相同的未知切片泛化能力。

### 2.3 官方示例有实际图像类型兼容问题，但不是当前人肝缓存的错误

本轮读取实际官方示例 patch：dtype={image['official_float_patch']['dtype']}，范围 {image['official_float_patch']['minimum']:.0f}–{image['official_float_patch']['maximum']:.0f}。
直接传新版 `ToPILImage` 会再乘 255 并回绕，**{100*image['official_float_patch']['changed_pixel_channel_fraction']:.2f}%** 的像素通道改变，平均绝对像素误差 **{image['official_float_patch']['mean_absolute_channel_error']:.2f}/255**。
先转 uint8 后与原像素完全一致。现有 SA/SP 复现已使用该修复。
四张人肝 TIFF 的 strip 数据范围完整，抽查 128 个缓存 patch 与坐标裁取的原图逐像素一致；当前人肝图像缓存没有重现这个错误。

### 2.4 生成式实验未完成，以及采样实现问题

GenAR HEG 上次日志停止在 epoch 39（已完成 40 轮），计划 60 轮的流水线没有跑完；之前 PPT 的“训练中”已过时。本轮复用这份可用权重完成推理。
根据完整 C1 验证集，在 top-k=1/2/8/32 中选择 **k={genar['top_k_expectation']} 的末级 token 期望解码**，中间尺度仍 top1。期望输出可以是小数，不能称为整数采样。
此选择没有按 D1 分数进行；k=1 对照也保留。权重仍按旧训练损失选出，本轮不能补称为验证 PCC 最优 checkpoint。
解码消融的 top1 对照来自与期望解码完全相同的前向 logits；不同 CPU/CUDA 路径可能导致接近并列的 token argmax 改变，不能把跨计算路径差异全归于解码算法。
GenAR top1 在 `log2(raw counts+1)` 尺度的 HEG200 PCC 为 0.2203，在面板相对表达尺度约 0.0688；这直接说明归一化目标改变会显著改变分数，不能混表。计数上限 2000 在 D1 影响约 0.042% 的元素（ALB 的 8.04% spots），需要披露，但对整体低 PCC 的影响未被证明是主因。

Stem 的原目标是 log 表达，旧推理默认图像裁剪范围 [-1,1] 不适用；旧选中 checkpoint 的 EMA 仍保留约 64% 随机初始化权重。
本轮在 HEG200 重训 **{stem_train['epochs_run']} 轮**，EMA 从 0.9999 改成 **0.99**，验证损失使用固定噪声/时间步，避免早停比较被随机验证噪声干扰。
C1 的预定 256-spot 子集比较 EMA/当前权重与裁剪选项，选择 **{stem['branch']} / {stem['clipping']}**，再以 DDIM100、3 次采样均值完成 C1/D1 推理。
选中 checkpoint 第 {stem_diag['epoch']} 轮，EMA 初始权重残留 {stem_diag['initial_ema_weight_fraction']:.3g}。
这是本地单卡适配，不是 UNI+CONCH 原论文配置复现。

### 2.5 MLP 不能冒充完整 ST-Net；流水线和统计代码也需修正

保留 MLP 为独立基线，新增 **DenseNet121 + 200 基因回归头** 的 ST-Net 适配；复用作者输出层初始化，ImageNet 预训练，微调全部权重，旋转/镜像增强，C1 PCC 早停。
最终 C1 选择 epoch={stnet['epoch']}，TTA={stnet['tta']}。训练/组织/目标与原论文不同，仍应标为适配。
修复 `scripts/44` 未定义变量 `HEG_RAW`；新的主评测入口为 `scripts/52`，会在缺方法时拒绝输出完整 benchmark。
流水线改为失败即停、每次独立日志，并补齐采样、解码、真实 ST-Net 和最终核验步骤。
PCC 常数列现在明确返回无定义；另修复 Moran's I 邻居列表重复去除 self 导致少一个近邻的问题。用独立 SciPy PCC、正负相关/常数和可手算 Moran 例子核验。

## 3. 当前完成的七法高表达基因对比

相同 A1+B1 训练、C1 验证、D1 测试，2,265 spots × 200 genes。所有输出逐项核对基因顺序、真值/坐标；来源 SHA256 见 JSON。
**主表真值不经过 PCA**。每个基因跨 spots 计算 PCC，再对固定 HEG200/HEG50 宏平均；不按测试相关性选最好的50个。
统一计分单位为 `log1p(10000 × 非负丰度 / HEG200丰度和)`，预测使用自己的总量，不借用测试真值的 library size。
这是面板内相对表达任务；训练目标/编码器/预算仍因方法而异，因此称“同任务适配比较”，不称完全控制预算的公平排名。

| 方法 | HEG200 PCC | HEG50 PCC | HEG200 Spearman | MAE | PCC 有效基因 |
|---|---:|---:|---:|---:|---:|
{table}

同一 HEG 面板、同一真值上的辅助对照：

| 对照版本 | HEG200 PCC | HEG50 PCC |
|---|---:|---:|
{ablations}

可以把 GenAR top1 与主表 GenAR 的差异归于当前验证选择的解码方案；BLEEP 两行可比较其适配流程。
不能把新旧基因面板的差值称为同一目标上的性能增益，也不能因某种改法在测试上没提升就偷偷换回分数更高的版本。

## 4. 辅助指标，明确分开

### 全转录组归一化的原始标签尺度

以下方法本身预测该尺度，可以直接与未重建真值比较。GenAR 当前输出原计数，缺少独立预测的全转录组 library size；不借用测试总量把它硬塞进此表。

| 方法 | HEG200 PCC | HEG50 PCC |
|---|---:|---:|
{native}

### 仅训练集拟合的 PCA50 重建诊断

仍使用相同 HEG200 和固定 HEG50，但真值和预测均投影回训练 PCA 空间；这些数值不是主表成绩，也不等同论文的联合 Harmony 口径。

| 方法 | PCA HEG200 PCC | PCA HEG50 PCC |
|---|---:|---:|
{pca}

## 5. 数据、论文与复核文件

- 30 个 ResSAT 原始 / HEST-IDC 文件的 SHA256 与既有来源清单全部一致；六个官方仓库 `git fsck` 通过。
- 本地七份论文 PDF 均可打开；ResSAT.pdf 为 41 页 2026 accepted manuscript，2024 文件为预印本，二者不混用。
- TIFF 检查覆盖所有声明数据段边界，128 个抽样 patch 精确匹配；这不声称重新逐字节下载了全部 TIFF。
- 另从 32 个图像块重新提取 ResNet18 特征：与缓存方向/尺度一致，CPU 与旧 CUDA 缓存的相对 L2 差约 0.1%；启用 CUDA TF32 后误差更小。属于数值近似一致，不声称逐位相等，见 `cached_feature_verification.json`。
- 当前所需论文、原始数据和预测可用，未重复下载已核验的大文件。GenAR/Stem 的论文基础模型特征和原训练权重不在官方代码包内，不能把本地轻量编码器适配称为完整论文复现。
- `results/verified_20260913/benchmark_predictions.npz`：统一表达尺度的七法预测、真值、barcode、基因与坐标。
- `benchmark.csv` / `benchmark_per_gene.csv`：总表与每个基因的 PCC。
- `fixed_HEG200_gene_audit.csv`：基因定义与仅训练选择的证据。
- `fresh_checkpoint_verification.json`：ResSAT 重新推理的 checkpoint 哈希与一致性。
- 原 11 页统一 Benchmark PPT 已在原文件上更新；第 7–8 页集中讲原因与复现。旧文件备份在结果目录内，避免根目录出现另一份容易混淆的“改进版”。

## 6. 当前能下的结论

高表达基因选择、评价口径混用、部分实现与未完成流水线是已证实的问题，均已给出修正和数据证据。
ResSAT 在原鼠脑设置下已验证接近论文的高表达基因结果；当前人肝任务的严格预测仍有明显差距。
四张人肝切片来自同一供者，不是独立患者测试。组织差异、切片差异、病理编码器缺失、训练规模/目标差异都是合理解释，但本轮没有受控实验证明“主要差距一定来自域偏移”。
下阶段应保留训练选定高表达基因协议，增加独立患者、匹配训练目标与编码器/计算预算，并做固定任务消融；不按测试分数挑基因或替换评价空间。

## 7. 复核命令

已有全局环境即可，无需新虚拟环境：

```powershell
wsl python3 scripts/50_audit_protocol_and_assets.py
wsl python3 scripts/50b_audit_image_alignment.py
wsl python3 scripts/51_verify_ressat_checkpoints.py
py -3.12 -X utf8 scripts/52_evaluate_verified_benchmark.py
py -3.12 -X utf8 scripts/53_update_verified_ppt.py
py -3.12 -X utf8 scripts/54_write_verified_report.py
```

训练和推理复跑见 `run_heg_pipeline.sh`。PPT 主报告没有把重算旧权重描述成新训练；本轮新训练的是 HEG Stem 与 DenseNet121 ST-Net。
'''
    (ROOT / 'docs/05_PCC_DIAGNOSIS.md').write_text(report, encoding='utf8')
    print('docs/05_PCC_DIAGNOSIS.md written')


if __name__ == '__main__':
    main()
