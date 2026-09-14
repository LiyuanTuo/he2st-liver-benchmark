"""修正两个 PPT 中的不规范用语（面板/口径/重建空间/域偏移/预算/冻结/macro 等）。

1. 按图片像素尺寸把已重新生成的、标签更清晰的四张图替换回 PPT；
2. 对全部页面做精确的用语替换（保留原有字体格式）。

运行：py -3.12 scripts/49_fix_wording.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from pptx import Presentation

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "figures/benchmark_ppt"

DECKS = [
    ROOT / "H&E空间转录组生成研究_统一Benchmark汇报_庹力元.pptx",
    ROOT / "H&E空间转录组生成研究_阶段汇报_庹力元.pptx",
]

# 像素尺寸 -> 重新生成的图片文件（两版 PPT 中各尺寸均只出现一次）
PICTURE_MAP = {
    (2368, 647): "cause_decomposition_v2.png",
    (2294, 777): "heg_benchmark_bars.png",
    (2257, 721): "heg_improvement_bars.png",
    (2268, 522): "heg_topgene_heatmaps.png",
}

REPLACEMENTS = [
    # ---- 统一 Benchmark 版 ----
    ("旧面板口径 · HEG 口径见第 8 页", "旧基因集与旧算法 · 新结果见第 8 页"),
    ("旧版混合口径表不再用于这张比较。", "旧版混合算法的表不再用于这张比较。"),
    ("图像编码器、训练预算不同，不是原论文方法的公平排名。",
     "图像编码器、训练投入不同，不是原论文方法的公平排名。"),
    ("同一个 ResSAT 实现已在官方数据上复现论文数值（top-50 HEG：0.880）；0.0x 来自三个协议差异叠加，而不是模型损坏。",
     "同一个 ResSAT 模型已在官方数据上复现论文结果（前 50 个高表达基因 PCC=0.880）；"
     "0.0x 是三个实验设置差异叠加造成的，不是模型写错。"),
    ("同一模型实现，仅换基因面板与评价口径：0.087 → 0.194（全 200 基因）→ 0.236（top-50 高表达基因）",
     "同一个模型，只换基因集与计算方式：0.087 → 0.194（200 个基因平均）→ 0.236（前 50 个高表达基因）"),
    ("① 基因面板选错", "① 基因挑错了"),
    ("旧 200 基因按方差选择；全转录组 top-50 高表达基因（ALB、HP…）重叠 0 个，低表达基因噪声主导平均。",
     "旧基准的 200 个基因是按波动大挑的，全转录组表达量最高的 50 个基因（ALB、HP…）一个都没选进来；"
     "低表达基因噪声大，把平均分拉低。"),
    ("论文对真值与预测同做 PCA/Harmony 重建后算 PCC；完整 log 空间含技术噪声。",
     "论文先把真值和预测做 PCA 去噪重建，再算 PCC；旧基准直接在原始对数表达值上算，混入了测量噪声。"),
    ("③ 划分与预算更硬", "③ 测试难度不同"),
    ("人肝跨切片 A/B→D 域偏移大；200 基因 + 冻结 ResNet18，训练预算低于论文。",
     "训练用 A、B 切片，测试用 D 切片，切片之间差异大；只训 200 个基因，图像编码器固定不动，"
     "训练投入远小于论文。"),
    ("改进：基准改为高表达基因（HEG）面板并重训", "改进：改按表达量挑选基因（高表达基因 HEG）并重新训练"),
    ("面板按训练集平均表达选 top-200 HEG（ALB、HP、SAA1…，零值率 1.4% vs 旧面板 24.7%）；已完成的四个方法全部提升，GenAR/Stem 训练中预留。",
     "新基准的 200 个基因按训练集平均表达量从高到低挑（ALB、HP、SAA1…），约 1.4% 的位置读数为 0"
     "（旧基因集 24.7%），噪声小得多。已完成的四个方法全部提升，GenAR/Stem 训练中预留。"),
    ("同一测试切片 C73_D1：旧面板 → HEG 面板（重建空间 top-50）",
     "同一张测试切片 C73_D1：旧基因集 → 高表达基因集（去噪后 · 前 50 个）"),
    ("已完成 · 四法重训", "已完成 · 四个方法重新训练"),
    ("（重建空间 top-50 HEG）", "（去噪后 · 前 50 个高表达基因）"),
    ("结论：面板与口径修正后四法全部显著提升；剩余差距主要来自跨切片域偏移。",
     "结论：换基因集、按论文方式计算后，四个方法全部明显提升；剩余差距主要因为训练切片与测试切片差异大。"),
    ("总排名仍应看第 6 页（旧面板）与第 8 页（HEG 面板）的全基因平均。",
     "总排名仍应看第 6 页（旧基因集）与第 8 页（高表达基因集）的全基因平均。"),
    ("新增 HEG 面板重训与论文口径重评；旧面板基准与官方案例结果保留。原版 PPT 保留；本次新增 11 页统一 Benchmark 汇报（GenAR/Stem 训练中，预留）。",
     "新增高表达基因集的重训与论文计算方式重评；旧基因集基准与官方案例结果全部保留。"
     "原版 PPT 保留；本次新增 11 页统一 Benchmark 汇报（GenAR/Stem 训练中，预留）。"),
    ("纠正旧表混用归一化口径的问题，六法重算。", "纠正旧表混用归一化算法的问题，六法重算。"),
    ("旧面板原空间：ResSAT 平均 PCC 0.087（六法最高），噪声主导、明显偏低。",
     "旧基准（原算法）：ResSAT 平均 PCC 0.087（六法最高），但明显偏低。"),
    ("改按论文口径（高表达基因 + PCA 重建空间）重选面板并重训后，top-50 HEG 平均 PCC 达 0.236。",
     "改按论文做法（挑高表达基因、先去噪再计算）重新训练后，前 50 个高表达基因平均 PCC 达 0.236。"),
    ("剩余差距主要来自跨切片划分与完整空间评价；GenAR/Stem 训练完成后补入对比。",
     "剩余差距主要因为训练切片与测试切片差异大；GenAR/Stem 训练完成后补入对比。"),
    ("下一步：补全 GenAR/Stem 重训 → 跨切片领域自适应 → 独立患者验证",
     "下一步：补全 GenAR/Stem → 减小切片间差异 → 独立患者验证"),
    # ---- 阶段汇报版 ----
    ("同一模型实现，只换基因面板与评价口径：全 200 基因平均 0.087 → 0.194，top-50 高表达基因 0.236",
     "同一个模型，只换基因集与计算方式：200 个基因平均 0.087 → 0.194，前 50 个高表达基因 0.236"),
    ("下图：仅按论文口径（PCA-50 重建空间 + top-50 HEG）重评旧预测，六法全部上移",
     "下图：仅换成论文的计算方式（先 PCA 去噪，再看前 50 个高表达基因），六个方法全部上升"),
    ("旧 200 基因按方差选择；全转录组 top-50 高表达基因（ALB、HP…）重叠 0 个。",
     "旧基准的 200 个基因是按波动大挑的，表达量最高的 50 个基因（ALB、HP…）一个都没选进来。"),
    ("跨切片 A/B→D 域偏移大；200 基因 + 冻结 ResNet18/轻量微调，预算低于论文。",
     "训练用 A、B 切片，测试用 D 切片，差异大；只训 200 个基因，图像编码器基本固定，训练投入远小于论文。"),
    ("HEG 面板：训练集平均表达 top-200（ALB、HP、SAA1…），零值率 1.4% vs 旧面板 24.7%",
     "新基因集：训练集平均表达量前 200（ALB、HP、SAA1…），约 1.4% 的位置读数为 0（旧基因集 24.7%）"),
    ("GenAR：HEG 面板训练中；Stem：排队中，完成后补入本表与图片。",
     "GenAR：高表达基因集训练中；Stem：排队中，完成后补入本表与图片。"),
    ("面板与口径修正后四法全部显著提升；剩余差距主要来自跨切片域偏移，下一步做领域自适应。",
     "换基因集、按论文方式计算后，四个方法全部明显提升；剩余差距主要因为切片之间差异大，"
     "下一步继续缩小切片差异。"),
    # ---- 阶段汇报版原有的少量不规范用语 ----
    ("同一 200 基因面板下比较", "同一套 200 个基因下比较"),
    ("作者预处理协议", "作者数据预处理方式"),
    ("全基因 macro PCC = 0.651", "全基因平均 PCC = 0.651"),
    ("固定人肝 200 genes、固定切片划分后 macro PCC = 0.100",
     "固定人肝 200 个基因、固定切片划分后平均 PCC = 0.100"),
    ("但不是固定面板上的总体性能", "但不是固定基因集上的总体性能"),
    ("严格固定 200 基因 macro PCC = 0.005", "严格固定 200 个基因的平均 PCC = 0.005"),
    ("预先固定的 200 基因 macro 指标", "预先固定的 200 个基因的平均指标"),
]


def replace_pictures(prs):
    replaced = 0
    for slide in prs.slides:
        for shape in slide.shapes:
            if shape.shape_type != 13:
                continue
            size = (shape.image.size[0], shape.image.size[1])
            if size in PICTURE_MAP:
                with open(ASSETS / PICTURE_MAP[size], "rb") as source:
                    new_bytes = source.read()
                # shape.part 是 SlidePart；真正的图片字节在 blip 对应的 ImagePart 里
                rId = shape._element.blip_rId
                image_part = slide.part.rels[rId].target_part
                image_part._blob = new_bytes
                replaced += 1
    return replaced


def replace_texts(prs):
    count = 0
    for slide in prs.slides:
        for shape in slide.shapes:
            if not shape.has_text_frame:
                continue
            for paragraph in shape.text_frame.paragraphs:
                for run in paragraph.runs:
                    for old, new in REPLACEMENTS:
                        if old in run.text:
                            run.text = run.text.replace(old, new)
                            count += 1
    return count


def replace_texts_paragraph_level(prs):
    """run 被打散（加粗高亮等原因）时，按整段拼接后替换，再合并为单个 run。"""
    count = 0
    for slide in prs.slides:
        for shape in slide.shapes:
            if not shape.has_text_frame:
                continue
            for paragraph in shape.text_frame.paragraphs:
                runs = paragraph.runs
                if not runs:
                    continue
                full = "".join(r.text for r in runs)
                new = full
                for old, new_s in REPLACEMENTS:
                    if old in new:
                        new = new.replace(old, new_s)
                if new != full:
                    runs[0].text = new
                    for r in runs[1:]:
                        r._r.getparent().remove(r._r)
                    count += 1
    return count


def main():
    for deck in DECKS:
        prs = Presentation(str(deck))
        pics = replace_pictures(prs)
        texts = replace_texts(prs)
        texts_para = replace_texts_paragraph_level(prs)
        prs.save(deck)
        print(f"{deck.name}: 图片替换 {pics} 处，run 级替换 {texts} 处，段落级替换 {texts_para} 处，页数 {len(prs.slides)}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
