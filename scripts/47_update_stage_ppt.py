"""把已完成的 HEG 改进结果插入《阶段汇报》PPT（原版风格）。

在"08 实验结果 · 空间热图检查"之后、"请老师批评指正"之前插入两页：
- 22：统一 benchmark 的 PCC 为什么只有 0.0x（三面板原因量化图 + 三张色条卡片）
- 23：改进：重选高表达基因面板（四法提升柱状图 + ALB 空间热图 + 状态卡片）

版式完全复刻原版：微软雅黑、B71C1C 标题、B08D57 金色分隔线、白色图片卡片
（D8D2C8 描边）、F7F4EF/3F5F89、EAF4EE/40765A、F8ECEB/B71C1C 色条卡片、
右上角页码。GenAR/Stem 训练中，以"进行中"卡片预留。

运行：py -3.12 scripts/47_update_stage_ppt.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_AUTO_SIZE, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
PPTX = ROOT / "H&E空间转录组生成研究_阶段汇报_庹力元.pptx"
ASSETS = ROOT / "figures/benchmark_ppt"
DATA = json.loads((ROOT / "results/heg_benchmark_interim.json").read_text(encoding="utf-8"))

RED, GOLD, INK, GRAY, BODY = "B71C1C", "B08D57", "262626", "666666", "404040"
CARD_BLUE = ("F7F4EF", "3F5F89")
CARD_GREEN = ("EAF4EE", "40765A")
CARD_RED = ("F8ECEB", "B71C1C")
OUTLINE = "D8D2C8"
FONT = "Microsoft YaHei"


def rect(slide, x, y, w, h, fill, line=None, rounded=False):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE,
        Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = RGBColor.from_string(fill)
    if line:
        shape.line.color.rgb = RGBColor.from_string(line)
        shape.line.width = Pt(0.8)
    else:
        shape.line.fill.background()
    if rounded:
        shape.adjustments[0] = 0.08
    return shape


def txt(slide, value, x, y, w, h, size=12, ink=BODY, bold=False, align=PP_ALIGN.LEFT):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = shape.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.auto_size = MSO_AUTO_SIZE.NONE
    frame.margin_left = frame.margin_right = Inches(.02)
    frame.margin_top = frame.margin_bottom = Inches(.01)
    for i, line in enumerate(str(value).split("\n")):
        p = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
        p.text = line
        p.font.name = FONT
        p.font.size = Pt(size)
        p.font.color.rgb = RGBColor.from_string(ink)
        p.font.bold = bold
        p.alignment = align
        p.space_before = Pt(0)
        p.space_after = Pt(2)
        p.line_spacing = 1.15
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


def header(slide, section, title, page):
    txt(slide, section, .55, .13, 2.4, .22, 13, GRAY)
    txt(slide, title, .55, .38, 11.9, .48, 24, RED, True)
    rect(slide, .55, .93, 12.22, .03, GOLD)
    txt(slide, page, 12.35, .16, .42, .22, 12, GRAY, align=PP_ALIGN.RIGHT)


def card(slide, x, y, w, h, accent, title, body):
    rect(slide, x, y, w, h, accent[0], OUTLINE, rounded=True)
    rect(slide, x, y, .06, h, accent[1])
    txt(slide, title, x + .18, y + .1, w - .32, .32, 15, INK, True)
    txt(slide, body, x + .18, y + .48, w - .32, h - .58, 11.5, BODY)


def build_cause_slide(prs):
    slide = insert_slide(prs, 14)
    header(slide, "08 实验结果", "统一 benchmark 的 PCC 为什么只有 0.0x？", "22")
    # 左侧白色图片卡片：两张量化图
    rect(slide, .54, 1.07, 8.3, 5.7, "FFFFFF", OUTLINE, rounded=True)
    picture(slide, ASSETS / "cause_decomposition_v2.png", .7, 1.3, 7.98, 2.2)
    txt(slide, "同一模型实现，只换基因面板与评价口径：全 200 基因平均 0.087 → 0.194，"
               "top-50 高表达基因 0.236",
        .78, 3.52, 7.8, .3, 11, BODY, True, PP_ALIGN.CENTER)
    picture(slide, ASSETS / "heg_benchmark_bars.png", 1.35, 3.86, 6.7, 2.32)
    txt(slide, "下图：仅按论文口径（PCA-50 重建空间 + top-50 HEG）重评旧预测，六法全部上移",
        .78, 6.22, 7.8, .42, 10.5, GRAY, align=PP_ALIGN.CENTER)
    # 右侧三张原因卡片
    card(slide, 9.08, 1.12, 3.55, 1.52, CARD_RED, "① 基因面板选错",
         "旧 200 基因按方差选择；全转录组 top-50 高表达基因（ALB、HP…）重叠 0 个。")
    card(slide, 9.08, 2.88, 3.55, 1.7, CARD_GREEN, "② 评价空间不同",
         "论文对真值与预测同做 PCA/Harmony 重建后算 PCC；完整 log 空间含技术噪声。")
    card(slide, 9.08, 4.82, 3.55, 1.55, CARD_BLUE, "③ 划分与预算更硬",
         "跨切片 A/B→D 域偏移大；200 基因 + 冻结 ResNet18/轻量微调，预算低于论文。")


def build_improvement_slide(prs):
    slide = insert_slide(prs, 15)
    header(slide, "08 实验结果", "改进：基准改为高表达基因（HEG）面板并重训", "23")
    rect(slide, .54, 1.07, 8.3, 5.7, "FFFFFF", OUTLINE, rounded=True)
    picture(slide, ASSETS / "heg_improvement_bars.png", .7, 1.28, 7.98, 2.5)
    txt(slide, "同一测试切片 C73_D1 · 已完成的四个方法全部提升（GenAR/Stem 训练中）",
        .78, 3.8, 7.8, .3, 11, BODY, True, PP_ALIGN.CENTER)
    picture(slide, ASSETS / "heg_topgene_heatmaps.png", .78, 4.18, 7.85, 1.72)
    txt(slide, "最高表达基因 ALB：实测真值与四法预测（统一色标，亮=表达高）",
        .78, 5.95, 7.8, .3, 11, BODY, True, PP_ALIGN.CENTER)
    txt(slide, "HEG 面板：训练集平均表达 top-200（ALB、HP、SAA1…），零值率 1.4% "
               "vs 旧面板 24.7%",
        .78, 6.28, 7.8, .38, 10.5, GRAY, align=PP_ALIGN.CENTER)
    # 右侧状态卡片
    t = DATA["available"]
    body = "\n".join(
        f"{name}：{t[name]['old_panel_raw_all200']:.3f} → {t[name]['heg_recon_top50']:.3f}"
        for name in ["ResSAT", "BLEEP", "Image Ridge", "MLP 基线"])
    card(slide, 9.08, 1.12, 3.55, 2.05, CARD_GREEN, "已完成 · 四法重训",
         body + "\n（重建空间 top-50 HEG）")
    card(slide, 9.08, 3.37, 3.55, 1.45, CARD_BLUE, "进行中 · 预留",
         "GenAR：HEG 面板训练中；Stem：排队中，完成后补入本表与图片。")
    card(slide, 9.08, 5.02, 3.55, 1.75, CARD_RED, "结论",
         "面板与口径修正后四法全部显著提升；剩余差距主要来自跨切片域偏移，"
         "下一步做领域自适应。")


def main():
    prs = Presentation(str(PPTX))
    assert len(prs.slides) == 15, f"期望 15 页，实际 {len(prs.slides)} 页"
    build_cause_slide(prs)
    build_improvement_slide(prs)
    assert len(prs.slides) == 17, f"期望 17 页，实际 {len(prs.slides)} 页"
    prs.save(PPTX)
    print(f"saved {PPTX.name}, slides={len(prs.slides)}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
