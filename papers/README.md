# 论文与数据来源

| 方法或资源 | 论文 | 作者代码 |
|---|---|---|
| ResSAT | [Genome Biology，2026](https://doi.org/10.1186/s13059-026-04168-x) | [ResSAT](https://github.com/Wonderangela123/ResSAT) |
| GenAR | [Medical Image Analysis，2026](https://doi.org/10.1016/j.media.2026.104232)；[预印本](https://arxiv.org/abs/2510.04315) | [genar](https://github.com/oyjr/genar) |
| Stem | [ICLR，2025](https://openreview.net/forum?id=FtjLUHyZAO) | [Stem](https://github.com/SichenZhu/Stem) |
| BLEEP | [NeurIPS，2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/df656d6ed77b565e8dcdfbf568aead0a-Abstract-Conference.html) | [BLEEP](https://github.com/bowang-lab/BLEEP) |
| ST-Net | [Nature Biomedical Engineering，2020](https://www.nature.com/articles/s41551-020-0578-x) | [ST-Net](https://github.com/bryanhe/ST-Net) |
| HEST-1k | [NeurIPS，2024](https://arxiv.org/abs/2406.16192) | [HEST](https://github.com/mahmoodlab/HEST) |
| 人肝单图块基线 | [BMC Bioinformatics，2026，27:168](https://doi.org/10.1186/s12859-026-06447-7) | [MAI-spatial-transcriptomics](https://github.com/KU-MedAI/MAI-spatial-transcriptomics) |

数据入口：[GSE240429](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE240429) · [BLEEP 表达与坐标](https://github.com/bowang-lab/BLEEP/tree/main/GSE240429_data) · [ResSAT 处理后示例](https://doi.org/10.5281/zenodo.20031209)。

## 本地文件

PDF 和文本索引仅保存在本地。2026-09-13 核验记录位于 `results/verified_20260913/paper_integrity.json`；人肝单图块论文的记录位于 `results/liver_context_20260913/additional_paper.json`。

| 本地 PDF | 版本 |
|---|---|
| `ResSAT.pdf` | 2026 接收稿，41 页 |
| `ResSAT_2024_preprint.pdf` | 2024 预印本，20 页 |
| `GenAR_2026_Medical_Image_Analysis_arXiv.pdf` | GenAR |
| `Stem_2025_ICLR.pdf` | Stem |
| `BLEEP_2023_NeurIPS.pdf` | BLEEP |
| `ST-Net_2020_Nature_Biomedical_Engineering.pdf` | ST-Net |
| `HEST-1k_2024_NeurIPS.pdf` | HEST-1k |
| `SinglePatch_Liver_Baseline_2026.pdf` | 人肝单图块基线，22 页 |

ResSAT 对照采用 2026 版本；2024 预印本仅作历史资料。GenAR 的论文特征和训练权重未随所用代码提供，Stem 原配置依赖 UNI/CONCH 特征。本项目的编码器替换及训练设置见[方法适配](../docs/02_METHODS_AND_PROTOCOL.md)，参考代码提交见 [THIRD_PARTY.md](../THIRD_PARTY.md)。
