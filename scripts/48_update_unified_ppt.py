"""把已完成的 HEG 改进结果插入《统一 Benchmark 汇报》PPT（9 页 → 11 页）。

在第 6 页（旧口径结果表）之后插入两页：
- 第 7 页：统一基准的 PCC 为什么只有 0.0x（三面板原因量化图 + 三张原因卡片）
- 第 8 页：改进：基准改为高表达基因面板并重训（四法提升柱状图 + ALB 空间热图 +
  状态卡片；GenAR/Stem 训练中以"进行中"预留）
并更新第 6 页提示、第 9/10 页示例的页码引用、第 11 页结论与全部页码。

版式复刻原版（27 页脚本风格）：白底、B71C1C 深红标题、B08D57 金色分隔线、
F7F4EF/EDF3F8/EDF5EF/F9EDEC 圆角卡片、底部来源行、右上角页码。

运行：py -3.12 scripts/48_update_unified_ppt.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_AUTO_SIZE, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
PPTX = ROOT / "H&E空间转录组生成研究_统一Benchmark汇报_庹力元.pptx"
ASSETS = ROOT / "figures/benchmark_ppt"
DATA = json.loads((ROOT / "results/heg_benchmark_interim.json").read_text(encoding="utf-8"))

RED, GOLD, INK, GRAY = "B71C1C", "B08D57", "262626", "666666"
LIGHT, BLUE, GREEN, PINK = "F7F4EF", "EDF3F8", "EDF5EF", "F9EDEC"
FONT = "Microsoft YaHei"


def rect(slide, x, y, w, h, fill=LIGHT, rounded=False):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE,
        Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = RGBColor.from_string(fill)
    shape.line.fill.background()
    if rounded:
        shape.adjustments[0] = .06
    return shape


def txt(slide, value, x, y, w, h, size=18, ink=INK, bold=False, align=PP_ALIGN.LEFT):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = shape.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.auto_size = MSO_AUTO_SIZE.NONE
    frame.margin_left = frame.margin_right = Inches(.015)
    frame.margin_top = frame.margin_bottom = Inches(.02)
    for i, line in enumerate(str(value).split("\n")):
        p = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
        p.text = line
        p.font.name = FONT
        p.font.size = Pt(size)
        p.font.color.rgb = RGBColor.from_string(ink)
        p.font.bold = bold
        p.alignment = align
        p.space_before = Pt(0)
        p.space_after = Pt(3)
        p.line_spacing = 1.12
    return shape


def picture(slide, path, x, y, w, h):
    with Image.open(path) as im:
        iw, ih = im.size
    scale = min(w / iw, h / ih)
    return slide.shapes.add_picture(
        str(path), Inches(x + (w - iw * scale) / 2), Inches(y + (h - ih * scale) / 2),
        width=Inches(iw * scale), height=Inches(ih * scale))


def insert_slide(prs, index):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    xml_slides = prs.slides._sldIdLst
    slides = list(xml_slides)
    xml_slides.remove(slides[-1])
    xml_slides.insert(index, slides[-1])
    return slide


def new_slide_style(prs, index, heading, subtitle, source, notes):
    slide = insert_slide(prs, index)
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = RGBColor.from_string("FFFFFF")
    txt(slide, heading, .48, .26, 12.25, .51, 27, RED, True)
    rect(slide, .5, .94, 12.3, .025, GOLD)
    txt(slide, subtitle, .51, 1.07, 12.24, .57, 16, GRAY)
    txt(slide, source, .5, 7.12, 11.86, .22, 9, GRAY)
    txt(slide, "00 / 11", 12.0, 7.11, .79, .24, 10, GRAY, align=PP_ALIGN.RIGHT)
    try:
        if slide.notes_slide is not None:
            slide.notes_slide.notes_text_frame.text = notes
    except Exception:
        pass
    return slide


def cause_card(slide, x, w, fill, heading, body):
    rect(slide, x, 5.6, w, 1.42, fill, rounded=True)
    txt(slide, heading, x + .18, 5.72, w - .36, .36, 17, RED, True)
    txt(slide, body, x + .18, 6.12, w - .36, .84, 12.5, INK)


def build_cause_slide(prs):
    slide = new_slide_style(
        prs, 6,
        "统一基准的 PCC 为什么只有 0.0x？",
        "同一个 ResSAT 实现已在官方数据上复现论文数值（top-50 HEG：0.880）；"
        "0.0x 来自三个协议差异叠加，而不是模型损坏。",
        "证据：results/ressat_official/evaluation.json；results/heg_benchmark_interim.json；"
        "data/processed/gse240429(_heg)/manifest.json",
        "三个原因各自有量化证据：① 旧 200 基因面板按方差选择，与全转录组 top-50 "
        "高表达基因零重叠；② 论文在 PCA/Harmony 重建空间评测，旧基准在完整 log 空间评含噪声真值；"
        "③ 旧基准是跨切片划分且各方法训练预算不同。改进结果见第 8 页。")
    picture(slide, ASSETS / "cause_decomposition_v2.png", .6, 1.82, 12.13, 3.31)
    txt(slide, "同一模型实现，仅换基因面板与评价口径：0.087 → 0.194（全 200 基因）"
               "→ 0.236（top-50 高表达基因）",
        .6, 5.2, 12.13, .34, 14.5, RED, True, PP_ALIGN.CENTER)
    cause_card(slide, .6, 3.95, PINK, "① 基因面板选错",
               "旧 200 基因按方差选择；全转录组 top-50 高表达基因（ALB、HP…）重叠 0 个，"
               "低表达基因噪声主导平均。")
    cause_card(slide, 4.7, 3.95, GREEN, "② 评价空间不同",
               "论文对真值与预测同做 PCA/Harmony 重建后算 PCC；完整 log 空间含技术噪声。")
    cause_card(slide, 8.8, 3.95, BLUE, "③ 划分与预算更硬",
               "人肝跨切片 A/B→D 域偏移大；200 基因 + 冻结 ResNet18，训练预算低于论文。")


def build_improvement_slide(prs):
    slide = new_slide_style(
        prs, 7,
        "改进：基准改为高表达基因（HEG）面板并重训",
        "面板按训练集平均表达选 top-200 HEG（ALB、HP、SAA1…，零值率 1.4% vs 旧面板 24.7%）；"
        "已完成的四个方法全部提升，GenAR/Stem 训练中预留。",
        "证据：results/heg_benchmark_interim.json；results/logs_heg_pipeline.txt；"
        "data/processed/gse240429_heg/manifest.json（基因列表 SHA256 已记录）",
        "面板重选只改基因选择标准（高表达代替高方差），划分、图像块、编码器特征、barcode 行序不变；"
        "训练集选 HEG 无测试泄漏。GenAR/Stem 完成后重跑 scripts/46、48 补入。")
    picture(slide, ASSETS / "heg_improvement_bars.png", .55, 1.82, 7.55, 2.4)
    txt(slide, "同一测试切片 C73_D1：旧面板 → HEG 面板（重建空间 top-50）",
        .6, 4.28, 7.45, .3, 12.5, RED, True, PP_ALIGN.CENTER)
    picture(slide, ASSETS / "heg_topgene_heatmaps.png", .55, 4.62, 7.55, 1.72)
    txt(slide, "最高表达基因 ALB：实测真值 vs 四法预测（统一色标，亮=表达高）",
        .6, 6.4, 7.45, .3, 12, GRAY, align=PP_ALIGN.CENTER)
    t = DATA["available"]
    body = "\n".join(
        f"{name}：{t[name]['old_panel_raw_all200']:.3f} → {t[name]['heg_recon_top50']:.3f}"
        for name in ["ResSAT", "BLEEP", "Image Ridge", "MLP 基线"])
    rect(slide, 8.25, 1.82, 4.53, 2.55, LIGHT, rounded=True)
    txt(slide, "已完成 · 四法重训", 8.44, 1.95, 4.15, .4, 17, RED, True)
    txt(slide, body + "\n（重建空间 top-50 HEG）", 8.44, 2.44, 4.15, 1.82, 13.5, INK)
    rect(slide, 8.25, 4.52, 4.53, 1.1, BLUE, rounded=True)
    txt(slide, "进行中 · 预留", 8.44, 4.63, 4.15, .34, 15.5, RED, True)
    txt(slide, "GenAR 训练中；Stem 排队，完成后补入。", 8.44, 5.02, 4.15, .55, 12.5, INK)
    txt(slide, "结论：面板与口径修正后四法全部显著提升；剩余差距主要来自跨切片域偏移。",
        8.44, 5.78, 4.3, 1.0, 13.5, RED, True)


def replace_in_slide(slide, old, new):
    replaced = False
    for shape in slide.shapes:
        if not shape.has_text_frame:
            continue
        for paragraph in shape.text_frame.paragraphs:
            for run in paragraph.runs:
                if old in run.text:
                    run.text = run.text.replace(old, new)
                    replaced = True
    return replaced


def replace_anywhere(prs, old, new):
    for slide in prs.slides:
        if replace_in_slide(slide, old, new):
            return True
    return False


def update_slide_six(prs):
    slide = prs.slides[5]
    for shape in slide.shapes:
        if shape.has_text_frame and "数值不代表原论文排名" in shape.text_frame.text:
            shape.text_frame.clear()
            p = shape.text_frame.paragraphs[0]
            p.text = "旧面板口径 · HEG 口径见第 8 页"
            p.font.name = FONT
            p.font.size = Pt(14)
            p.font.color.rgb = RGBColor.from_string(GRAY)
            p.alignment = PP_ALIGN.RIGHT
            return


def update_conclusion(prs):
    slide = prs.slides[-1]
    t = DATA["available"]["ResSAT"]
    for shape in slide.shapes:
        if not shape.has_text_frame:
            continue
        text = shape.text_frame.text
        if text.startswith("ResSAT 在本次适配中平均 PCC 最高"):
            frame = shape.text_frame
            frame.clear()
            paragraphs = [
                f"旧面板原空间：ResSAT 平均 PCC {t['old_panel_raw_all200']:.3f}（六法最高），噪声主导、明显偏低。",
                f"改按论文口径（高表达基因 + PCA 重建空间）重选面板并重训后，"
                f"top-50 HEG 平均 PCC 达 {t['heg_recon_top50']:.3f}。",
                "剩余差距主要来自跨切片划分与完整空间评价；GenAR/Stem 训练完成后补入对比。",
            ]
            for i, line in enumerate(paragraphs):
                p = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
                p.text = line
                p.font.name = FONT
                p.font.size = Pt(17)
                p.font.color.rgb = RGBColor.from_string(INK)
                p.space_before = Pt(0)
                p.space_after = Pt(4)
                p.line_spacing = 1.12
            return


def renumber(prs):
    total = len(prs.slides)
    pattern = re.compile(r"^\d{2} / \d{2}$")
    for index, slide in enumerate(prs.slides, start=1):
        for shape in slide.shapes:
            if shape.has_text_frame and pattern.match(shape.text_frame.text.strip()):
                shape.text_frame.clear()
                p = shape.text_frame.paragraphs[0]
                p.text = f"{index:02d} / {total:02d}"
                p.font.name = FONT
                p.font.size = Pt(10)
                p.font.color.rgb = RGBColor.from_string(GRAY)
                p.alignment = PP_ALIGN.RIGHT


def main():
    prs = Presentation(str(PPTX))
    assert len(prs.slides) == 9, f"期望 9 页旧 PPT，实际 {len(prs.slides)} 页"
    build_cause_slide(prs)
    build_improvement_slide(prs)
    update_slide_six(prs)
    replace_anywhere(prs, "总排名仍应看第 6 页的全基因平均。",
                     "总排名仍应看第 6 页（旧面板）与第 8 页（HEG 面板）的全基因平均。")
    replace_anywhere(prs,
                     "所有重新评测均复用现有权重与预测；未新增训练。原版 PPT 保留；本次新增 9 页统一 Benchmark 汇报",
                     "新增 HEG 面板重训与论文口径重评；旧面板基准与官方案例结果保留。"
                     "原版 PPT 保留；本次新增 11 页统一 Benchmark 汇报（GenAR/Stem 训练中，预留）")
    replace_anywhere(prs, "下一步：统一编码器与训练预算 → 排查生成式适配 → 增加独立患者测试",
                     "下一步：补全 GenAR/Stem 重训 → 跨切片领域自适应 → 独立患者验证")
    update_conclusion(prs)
    renumber(prs)
    assert len(prs.slides) == 11, f"期望 11 页，实际 {len(prs.slides)} 页"
    prs.core_properties.title = "H&E 空间转录组生成研究：统一 Benchmark 汇报（HEG 改进版）"
    prs.save(PPTX)
    print(f"saved {PPTX.name}, slides={len(prs.slides)}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
