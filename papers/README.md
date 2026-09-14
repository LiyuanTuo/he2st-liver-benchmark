# 论文与代码来源清单

初次下载日期：2026-08-30。2026-09-13 已验证七份 PDF 均可解析，页数及 SHA256 见 `../results/verified_20260913/paper_integrity.json`。PDF 文本索引位于 `papers/text/`，仅用于本地检索。

2026-09-13 续研另补充同一人肝数据的2026年基线论文，22页全部解析；下载来源、大小与SHA256见 `../results/liver_context_20260913/additional_paper.json`。它的HEG PCC约0.310，但采用slide3测试、BLEEP/Harmony基因流程，不能与本项目D1固定面板直接等同比较。

| 方法/资源 | 本地文件 | 正式来源 | 官方代码 | 当前可复现性 |
|---|---|---|---|---|
| ResSAT | `ResSAT.pdf`（2026 接收稿，41 页）；`ResSAT_2024_preprint.pdf`（20 页） | Genome Biology 2026, DOI 10.1186/s13059-026-04168-x；主结果以 2026 接收稿 Tables 1–2 为准 | 正式版：`Wonderangela123/ResSAT`；Zenodo `10.5281/zenodo.20031323` | 代码、处理后示例及本地 SA/SP 权重可用；完整原始预处理脚本仍未提供 |
| GenAR | `GenAR_2026_Medical_Image_Analysis_arXiv.pdf` | Medical Image Analysis 2026, DOI 10.1016/j.media.2026.104232 | `oyjr/genar` | 模型/训练代码可用；没有论文特征、checkpoint；论文配置使用 H100 80 GB |
| Stem | `Stem_2025_ICLR.pdf` | ICLR 2025 | `SichenZhu/Stem` | 代码可用；需要 HEST 数据和 UNI/CONCH 特征/权重 |
| BLEEP | `BLEEP_2023_NeurIPS.pdf` | NeurIPS 2023 | `bowang-lab/BLEEP` | 代码和 GSE240429 表达/坐标可用；H&E 需从 GEO 下载 |
| ST-Net | `ST-Net_2020_Nature_Biomedical_Engineering.pdf` | Nature Biomedical Engineering 2020 | `bryanhe/ST-Net` | 经典基线代码可用，但环境较旧 |
| HEST-1k | `HEST-1k_2024_NeurIPS.pdf` | NeurIPS 2024 | `mahmoodlab/HEST` | 数据可按样本下载；完整集合很大，本项目只下载所需子集 |
| 同人肝单图块基线 | `SinglePatch_Liver_Baseline_2026.pdf`（22页） | [BMC Bioinformatics 2026, 27:168](https://link.springer.com/article/10.1186/s12859-026-06447-7) | [KU-MedAI/MAI-spatial-transcriptomics](https://github.com/KU-MedAI/MAI-spatial-transcriptomics) | 论文已下载并核验；本轮用作同数据背景证据，未声称复现其EfficientNet结果 |

## 已保留的官方代码

官方仓库作为只读参照放在 `third_party/`。实验适配代码放在项目自己的 `scripts/` 和后续 `src/`，避免混淆“作者代码”和“本项目修改”。

| 本地目录 | commit |
|---|---|
| `third_party/GenAR` | `223a0a6` |
| `third_party/Stem` | `cbc3c7c` |
| `third_party/ST-Net` | `43022c1` |
| `third_party/BLEEP` | `2395967` |
| `third_party/HEST` | `3ddb5ea` |
| `third_party/ResSAT` | `10d60ab` |

## 重要审计结论

1. ResSAT 2024 预印本/PMC 页面仍指向已失效的 `Wonderangela/ResSAT`，2026 正式版已改为可访问的 `Wonderangela123/ResSAT`，并在 Zenodo 存档。必须以正式版链接为准。
2. ResSAT 2024 预印本与 2026 正式版的方法/实验范围有明显变化；正式版新增 Fourier 坐标编码、FiLM 融合、更多组织数据及嵌套交叉验证。本地预印本只用于追溯。
3. **ResSAT 官方示例数据与新版 torchvision 不兼容（本项目实测）**：示例 `dataset.pkl` 的 patch 为 float32(0–255)，而新版 `ToPILImage` 对浮点数组按 [0,1] 缩放，导致 ×255 后按 uint8 回绕、图像被静默损坏；修复方式为加载后转 uint8（见 docs/02 审计发现 A）。
4. GenAR 官方仓库明确不提供训练所用预计算图像特征、checkpoint 和数据文件。论文附录列出基因面板，但要复现原数值还需相同切片版本、特征和预处理。
5. GenAR 主配置使用 4 张 H100 80 GB。8 GB 消费级 GPU 可以验证模型机制和统一小型实验，但不能无说明地等同于论文计算规模。
6. Stem 需要病理基础模型特征；UNI/CONCH 的模型条款、鉴权和权重许可与代码许可证分开。官方训练脚本硬编码 UNI+CONCH 拼接与 `cuda:6`/DDP，单卡适配需替换数据管线（本项目 scripts/13/14）。

## 链接

- ResSAT 正式文章：https://link.springer.com/article/10.1186/s13059-026-04168-x
- ResSAT 正式代码：https://github.com/Wonderangela123/ResSAT
- ResSAT 处理后示例数据：https://doi.org/10.5281/zenodo.20031209
- GenAR：https://arxiv.org/abs/2510.04315
- Stem：https://openreview.net/forum?id=FtjLUHyZAO
- BLEEP：https://proceedings.neurips.cc/paper_files/paper/2023/hash/df656d6ed77b565e8dcdfbf568aead0a-Abstract-Conference.html
- ST-Net：https://www.nature.com/articles/s41551-020-0578-x
- HEST-1k：https://arxiv.org/abs/2406.16192
- GSE240429：https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE240429
