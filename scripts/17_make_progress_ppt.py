"""生成 H&E→空间转录组阶段汇报 PPT（无需命令行参数）。"""

from pathlib import Path
import gzip
import json

from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE, MSO_CONNECTOR
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.util import Inches, Pt


ROOT = Path(r"D:\Code\H&E Generation")
ASSET = ROOT / "figures" / "ppt_assets"
OUT = ROOT / "H&E空间转录组生成研究_阶段汇报_庹力元.pptx"

W, H = Inches(13.333333), Inches(7.5)
RED = "B71C1C"
RED2 = "C7473E"
GOLD = "B08D57"
INK = "262626"
MUTED = "666666"
LIGHT = "F7F4EF"
BLUE = "EAF2F8"
GREEN = "EAF4EE"
PINK = "F8ECEB"
YELLOW = "F7F1E3"
PURPLE = "F2ECF7"
WHITE = "FFFFFF"
LINE = "D8D2C8"
NAVY = "3F5F89"
SKY = "91BCD2"


def rgb(value):
    value = value.lstrip("#")
    return RGBColor(int(value[:2], 16), int(value[2:4], 16), int(value[4:], 16))


def prepare_assets():
    ASSET.mkdir(parents=True, exist_ok=True)

    crops = {
        "stnet_arch.png": (ASSET / "stnet_page2.png", (215, 120, 1290, 450)),
        "ressat_arch.png": (ASSET / "ressat_page25.png", (210, 105, 1300, 770)),
        "genar_arch.png": (ASSET / "genar_page4.png", (220, 145, 1210, 815)),
        "bleep_arch.png": (ROOT / "third_party" / "BLEEP" / "BLEEP_overview.png", (0, 0, 838, 650)),
    }
    for name, (src, box) in crops.items():
        dst = ASSET / name
        with Image.open(src) as im:
            im.crop(box).convert("RGB").save(dst, quality=95)

    # Stem 官方图片带透明背景；合成到白底后放入 PPT，避免部分查看器把透明区显示成黑色。
    with Image.open(ROOT / "third_party" / "Stem" / "assets" / "fig1.png") as im:
        rgba = im.convert("RGBA")
        white = Image.new("RGBA", rgba.size, "white")
        white.alpha_composite(rgba)
        white.convert("RGB").save(ASSET / "stem_arch.png", quality=95)

    # 四张 GSE240429 低分辨率 H&E 缩略图，来自本项目实际下载文件。
    panels = []
    labels = [("C73_A1", 2378), ("C73_B1", 2349), ("C73_C1", 2277), ("C73_D1", 2265)]
    raw = ROOT / "data" / "raw" / "GSE240429"
    for sample, spots in labels:
        suffix = sample.replace("_", "")
        matches = list(raw.glob(f"*_{suffix}_tissue_lowres_image.png.gz"))
        if not matches:
            matches = list(raw.glob(f"*_{sample}_tissue_lowres_image.png.gz"))
        with gzip.open(matches[0], "rb") as stream:
            im = Image.open(stream).convert("RGB")
            im.thumbnail((560, 360), Image.Resampling.LANCZOS)
            panel = Image.new("RGB", (590, 420), "white")
            panel.paste(im, ((590 - im.width) // 2, 35 + (350 - im.height) // 2))
            draw = ImageDraw.Draw(panel)
            try:
                font = ImageFont.truetype("arial.ttf", 25)
            except OSError:
                font = ImageFont.load_default()
            draw.text((16, 5), f"{sample}  |  {spots:,} spots", fill="#262626", font=font)
            panels.append(panel)
    montage = Image.new("RGB", (1180, 840), "white")
    for i, panel in enumerate(panels):
        montage.paste(panel, ((i % 2) * 590, (i // 2) * 420))
    montage.save(ASSET / "gse240429_four_slides.png", quality=95)

    # 从纵向定性结果中选出真值、ResSAT、GenAR、Stem 四行，便于横向汇报。
    src = Image.open(ROOT / "figures" / "05_gene_spatial_predictions.png").convert("RGB")
    bands = [(0, 400), (1680, 2200), (2160, 2680), (2640, src.height)]
    selected = [src.crop((0, y1, src.width, y2)) for y1, y2 in bands]
    canvas = Image.new("RGB", (src.width, sum(x.height for x in selected)), "white")
    y = 0
    for band in selected:
        canvas.paste(band, (0, y))
        y += band.height
    canvas.save(ASSET / "qualitative_selected.png", quality=95)


def blank(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    bg = slide.background.fill
    bg.solid()
    bg.fore_color.rgb = rgb(WHITE)
    return slide


def box(slide, x, y, w, h, fill=WHITE, line=LINE, radius=True, width=1):
    kind = MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE if radius else MSO_AUTO_SHAPE_TYPE.RECTANGLE
    shape = slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = rgb(fill)
    shape.line.color.rgb = rgb(line)
    shape.line.width = Pt(width)
    if radius:
        try:
            shape.adjustments[0] = 0.08
        except Exception:
            pass
    return shape


def text(slide, value, x, y, w, h, size=16, color=INK, bold=False,
         font="Microsoft YaHei", align=PP_ALIGN.LEFT, valign=MSO_ANCHOR.TOP,
         margin=0.05, line_spacing=1.05):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = shape.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.auto_size = MSO_AUTO_SIZE.TEXT_TO_FIT_SHAPE
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = Inches(margin)
    tf.vertical_anchor = valign
    lines = str(value).split("\n")
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.font.name = font
        p.font.size = Pt(size)
        p.font.bold = bold
        p.font.color.rgb = rgb(color)
        p.alignment = align
        p.space_after = Pt(0)
        p.space_before = Pt(0)
        p.line_spacing = line_spacing
    return shape


def rich(slide, runs, x, y, w, h, size=15, color=INK, align=PP_ALIGN.LEFT):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = shape.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.auto_size = MSO_AUTO_SIZE.TEXT_TO_FIT_SHAPE
    tf.margin_left = tf.margin_right = Inches(0.06)
    tf.margin_top = tf.margin_bottom = Inches(0.04)
    p = tf.paragraphs[0]
    p.alignment = align
    for value, bold, run_color in runs:
        r = p.add_run()
        r.text = value
        r.font.name = "Microsoft YaHei"
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.color.rgb = rgb(run_color or color)
    return shape


def title(slide, value, page, kicker=None):
    if kicker:
        text(slide, kicker, 0.55, 0.13, 2.4, 0.22, 8.5, RED, True)
    text(slide, value, 0.55, 0.38, 11.9, 0.48, 24, RED, True)
    rule = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.RECTANGLE, Inches(0.55), Inches(0.93), Inches(12.22), Inches(0.025))
    rule.fill.solid(); rule.fill.fore_color.rgb = rgb(GOLD); rule.line.fill.background()
    text(slide, f"{page:02d}", 12.35, 0.16, 0.42, 0.22, 9, MUTED, True, align=PP_ALIGN.RIGHT)


def footer(slide, value):
    text(slide, value, 0.56, 7.18, 12.15, 0.20, 7.3, MUTED, False, margin=0)


def section_tag(slide, value, x, y, w=1.45, fill=RED):
    shape = box(slide, x, y, w, 0.28, fill, fill, True, 0)
    text(slide, value, x + .03, y + .015, w - .06, .22, 9, WHITE, True, align=PP_ALIGN.CENTER, valign=MSO_ANCHOR.MIDDLE)
    return shape


def card(slide, heading, body, x, y, w, h, fill=LIGHT, accent=RED, body_size=13, head_size=14):
    box(slide, x, y, w, h, fill, LINE, True, 0.8)
    slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.RECTANGLE, Inches(x), Inches(y), Inches(0.055), Inches(h)).fill.solid()
    bar = slide.shapes[-1]
    bar.fill.fore_color.rgb = rgb(accent); bar.line.fill.background()
    text(slide, heading, x + .18, y + .12, w - .32, .32, head_size, accent, True)
    text(slide, body, x + .18, y + .52, w - .34, h - .62, body_size, INK, False, line_spacing=1.08)


def add_picture_contain(slide, path, x, y, w, h, line=None):
    with Image.open(path) as im:
        iw, ih = im.size
    scale = min(w / iw, h / ih)
    pw, ph = iw * scale, ih * scale
    pic = slide.shapes.add_picture(str(path), Inches(x + (w - pw) / 2), Inches(y + (h - ph) / 2), Inches(pw), Inches(ph))
    if line:
        pic.line.color.rgb = rgb(line)
        pic.line.width = Pt(0.7)
    return pic


def arrow(slide, x1, y1, x2, y2, color=RED, width=2):
    ln = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    ln.line.color.rgb = rgb(color)
    ln.line.width = Pt(width)
    ln.line.end_arrowhead = True
    return ln


def number_badge(slide, n, x, y, fill=RED):
    s = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.OVAL, Inches(x), Inches(y), Inches(.31), Inches(.31))
    s.fill.solid(); s.fill.fore_color.rgb = rgb(fill); s.line.fill.background()
    text(slide, str(n), x, y + .005, .31, .25, 9.5, WHITE, True, align=PP_ALIGN.CENTER, valign=MSO_ANCHOR.MIDDLE)


def add_method_step(slide, n, heading, body, x, y, w, fill=LIGHT):
    box(slide, x, y, w, .78, fill, LINE, True, .7)
    number_badge(slide, n, x + .12, y + .13)
    text(slide, heading, x + .52, y + .10, w - .64, .25, 11.5, RED, True)
    text(slide, body, x + .52, y + .37, w - .64, .30, 9.6, INK)


def add_table(slide, data, x, y, w, h, col_widths=None, header_fill=RED, font_size=9.5):
    rows, cols = len(data), len(data[0])
    table = slide.shapes.add_table(rows, cols, Inches(x), Inches(y), Inches(w), Inches(h)).table
    if col_widths:
        for col, width in zip(table.columns, col_widths):
            col.width = Inches(width)
    for r in range(rows):
        for c in range(cols):
            cell = table.cell(r, c)
            cell.text = str(data[r][c])
            cell.margin_left = cell.margin_right = Inches(.05)
            cell.margin_top = cell.margin_bottom = Inches(.035)
            cell.fill.solid()
            cell.fill.fore_color.rgb = rgb(header_fill if r == 0 else (WHITE if r % 2 else LIGHT))
            for p in cell.text_frame.paragraphs:
                p.font.name = "Microsoft YaHei"
                p.font.size = Pt(font_size if r else font_size + .2)
                p.font.bold = r == 0
                p.font.color.rgb = rgb(WHITE if r == 0 else INK)
                p.alignment = PP_ALIGN.CENTER if c > 0 else PP_ALIGN.LEFT
                p.space_after = Pt(0)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    return table


def build_deck():
    prepare_assets()
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H

    # 1 封面
    s = blank(prs)
    s.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.RECTANGLE, 0, 0, Inches(.18), H).fill.solid()
    s.shapes[-1].fill.fore_color.rgb = rgb(RED); s.shapes[-1].line.fill.background()
    text(s, "研究阶段汇报", .68, .54, 2.2, .32, 11, RED, True)
    text(s, "基于 H&E 染色图像的\n空间转录组数据生成研究", .68, 1.16, 7.4, 1.35, 31, INK, True)
    rich(s, [("候选方法 · 原始数据集 · 统一 Benchmark · 当前实验结果", False, MUTED)], .72, 2.68, 8.2, .38, 14)
    add_picture_contain(s, ROOT / "figures" / "03_he_and_st_task_example.png", 7.55, .68, 5.35, 4.35)
    box(s, .72, 4.55, 6.0, 1.45, PINK, PINK, True, 0)
    text(s, "核心问题", .98, 4.80, 1.2, .28, 12, RED, True)
    text(s, "能否只利用常规 H&E 组织图像，推断每个空间位置的多基因表达谱？", .98, 5.14, 5.35, .62, 17, INK, True)
    text(s, "庹力元 · 东南大学", .72, 6.72, 4.5, .30, 12, INK, True)
    text(s, "2026 年 9 月", 10.55, 6.72, 2.3, .30, 11, MUTED, False, align=PP_ALIGN.RIGHT)

    # 2 任务形式
    s = blank(prs); title(s, "任务是什么：从组织形态预测每个 spot 的表达向量", 2, "01 研究问题")
    add_picture_contain(s, ROOT / "figures" / "03_he_and_st_task_example.png", .58, 1.12, 12.15, 3.38)
    for i, (hd, bd, fill) in enumerate([
        ("输入", "整张 H&E 图像 + 每个测量点坐标", BLUE),
        ("模型看到的样本", "以 spot 为中心裁出 224×224 图像块", YELLOW),
        ("输出", "每个 spot 对应一个多基因表达向量", GREEN),
        ("最终数据形态", "所有 spot × 所有目标基因的矩阵", PURPLE),
    ]):
        card(s, hd, bd, .70 + i * 3.12, 4.68, 2.84, 1.27, fill, [NAVY, GOLD, "40765A", "6B4D8A"][i], 11.5, 12.5)
    box(s, .70, 6.18, 12.0, .65, PINK, PINK, True, 0)
    rich(s, [("关键澄清：", True, RED), ("模型不是“生成一张彩色热图”；它先生成数值矩阵，热图只是把某一列基因表达画回组织坐标。", False, INK)], .92, 6.36, 11.55, .27, 12.3)
    footer(s, "示例均由本项目真实 GSE240429 人肝 Visium 数据生成；C73_A1 含 2,378 个组织内 spot。")

    # 3 数据长什么样
    s = blank(prs); title(s, "空间转录组数据长什么样：一张切片对应很多张基因空间图", 3, "01 研究问题")
    add_picture_contain(s, ROOT / "figures" / "01_spatial_transcriptomics_examples.png", .55, 1.08, 12.2, 3.02)
    box(s, .60, 4.33, 5.65, 2.35, LIGHT, LINE, True, .7)
    add_picture_contain(s, ROOT / "figures" / "02_expression_matrix_example.png", .76, 4.48, 5.34, 1.94)
    card(s, "如何读图", "• 同一个点有 200 个目标基因数值\n• 同一个基因在 2,378 个点上形成一张空间图\n• GLUL、CYP1A2 显示肝小叶分区；IGKC 更稀疏\n• 模型要同时恢复数值强弱与空间结构", 6.52, 4.31, 6.18, 2.36, PINK, RED, 12.2, 14)
    footer(s, "表达值为每个 spot 总量归一化至 10,000 后 log1p；矩阵图仅展示 180 个 spot × 40 个基因的小窗口。")

    # 4 方法谱系
    s = blank(prs); title(s, "待选方法不是同一种模型：五条路线代表五种建模假设", 4, "02 候选论文")
    methods = [
        ("2020", "ST-Net", "直接回归", "图像块 → 连续表达", BLUE),
        ("2023", "BLEEP", "对比检索", "图像嵌入 → 邻居表达", YELLOW),
        ("2025", "Stem", "条件扩散", "噪声 → 多样表达样本", PURPLE),
        ("2026", "ResSAT", "空间交互", "图像 + 坐标 + spot 注意力", GREEN),
        ("2026", "GenAR", "自回归生成", "粗到细生成离散 counts", PINK),
    ]
    for i, (year, name, family, desc, fill) in enumerate(methods):
        x = .64 + i * 2.50
        text(s, year, x, 1.22, 2.12, .30, 11, MUTED, True, align=PP_ALIGN.CENTER)
        box(s, x, 1.65, 2.12, 2.62, fill, LINE, True, .8)
        text(s, name, x + .12, 1.89, 1.88, .35, 17, RED, True, align=PP_ALIGN.CENTER)
        text(s, family, x + .18, 2.42, 1.76, .30, 12, INK, True, align=PP_ALIGN.CENTER)
        text(s, desc, x + .22, 2.94, 1.68, .70, 10.5, INK, False, align=PP_ALIGN.CENTER, valign=MSO_ANCHOR.MIDDLE)
        if i < 4:
            arrow(s, x + 2.13, 2.93, x + 2.43, 2.93, GOLD, 1.5)
    card(s, "选择逻辑", "保留一个经典回归基线（ST-Net）、一个检索式方法（BLEEP）、一个显式空间交互模型（ResSAT），再比较两种真正的生成式路线（Stem 与 GenAR）。", .68, 4.70, 5.95, 1.38, LIGHT, RED, 12.3, 14)
    card(s, "资源而非预测模型", "HEST-1k 用于扩展 benchmark 与跨组织评测。它是标准化数据集/工具库，不应与上述五个预测模型并列比较准确率。", 6.86, 4.70, 5.78, 1.38, LIGHT, NAVY, 12.3, 14)
    box(s, .68, 6.28, 11.96, .55, PINK, PINK, True, 0)
    text(s, "本项目当前重点：先在同一人肝数据、同一切片划分、同一 200 基因面板下比较；再扩展到 HEST 多组织。", .92, 6.45, 11.45, .24, 12, RED, True, align=PP_ALIGN.CENTER)
    footer(s, "候选论文：ST-Net (Nature Biomedical Engineering 2020)、BLEEP (NeurIPS 2023)、Stem (ICLR 2025)、ResSAT (Genome Biology 2026)、GenAR (Medical Image Analysis 2026)。")

    # 5 ST-Net
    s = blank(prs); title(s, "ST-Net：把每个 spot 当成独立图像回归样本", 5, "02 候选论文")
    box(s, .58, 1.10, 7.28, 3.15, WHITE, LINE, True, .8)
    add_picture_contain(s, ASSET / "stnet_arch.png", .72, 1.22, 7.00, 2.78)
    add_method_step(s, 1, "裁图", "以空间测量点为中心取 224×224 patch", 8.10, 1.11, 4.55, BLUE)
    add_method_step(s, 2, "共享卷积编码", "120 个卷积层为所有基因提取同一组 1,024 维形态特征", 8.10, 2.00, 4.55, BLUE)
    add_method_step(s, 3, "多输出回归", "全连接层一次输出 250 个基因的 log-normalized 表达", 8.10, 2.89, 4.55, BLUE)
    card(s, "原论文数据", "68 张乳腺癌切片 / 23 位患者\n30,612 个空间 spot\nleave-one-patient-out：22 人训练，1 人测试", .62, 4.55, 3.80, 1.66, LIGHT, RED, 12, 13.5)
    card(s, "原论文结论", "250 个基因中，102 个基因在至少 20/23 位患者上呈一致正相关；外部 10x 乳腺切片上 234 个基因平均 PCC=0.33。", 4.60, 4.55, 4.10, 1.66, LIGHT, NAVY, 11.5, 13.5)
    card(s, "局限", "每个 spot 独立预测，不显式利用邻域；连续回归容易生成过度平滑的表达值，难以恢复零值与多样性。", 8.88, 4.55, 3.78, 1.66, LIGHT, GOLD, 12, 13.5)
    footer(s, "图源：He et al., Nature Biomedical Engineering 4, 827–834 (2020), doi:10.1038/s41551-020-0578-x。")

    # 6 BLEEP
    s = blank(prs); title(s, "BLEEP：不直接回归，而是在联合空间中检索参考表达", 6, "02 候选论文")
    box(s, .55, 1.08, 6.35, 4.72, WHITE, LINE, True, .8)
    add_picture_contain(s, ASSET / "bleep_arch.png", .70, 1.20, 6.05, 4.40)
    add_method_step(s, 1, "双模态编码", "ResNet50 编码图像；MLP 编码表达", 7.18, 1.10, 5.45, YELLOW)
    add_method_step(s, 2, "对比学习", "配对的图像/表达靠近，不配对样本远离，得到 256 维联合空间", 7.18, 2.00, 5.45, YELLOW)
    add_method_step(s, 3, "KNN 推断", "测试图像找到 K=50 个最近训练表达向量，取均值作为预测", 7.18, 2.90, 5.45, YELLOW)
    card(s, "原论文数据就是本项目 benchmark 来源", "4 张连续 16 μm 人肝切片；2,378 / 2,349 / 2,277 / 2,265 spots，共 9,269 spots；10x Visium；HVG 并集 3,467 基因。", 7.18, 3.83, 5.45, 1.63, LIGHT, RED, 11.5, 13)
    box(s, 7.18, 5.72, 5.45, .82, PINK, PINK, True, 0)
    rich(s, [("方法特点：", True, RED), ("训练稳定、推断直观，但输出受训练库覆盖范围限制；K 邻居平均也会造成平滑。", False, INK)], 7.38, 5.95, 5.06, .38, 11.5)
    footer(s, "图源：Xie et al., NeurIPS 2023；数据：GSE240429。原论文使用 4×V100、batch 512、150 epochs。")

    # 7 ResSAT 方法
    s = blank(prs); title(s, "ResSAT：图像、坐标与 spot 间交互共同预测表达", 7, "02 候选论文")
    box(s, .55, 1.08, 8.05, 4.62, WHITE, LINE, True, .8)
    add_picture_contain(s, ASSET / "ressat_arch.png", .68, 1.20, 7.80, 4.35)
    steps = [
        ("图像支路", "ResNet50 → 512 维图像特征"),
        ("空间支路", "坐标经随机 Fourier 映射 → 128 维"),
        ("FiLM 融合", "坐标生成缩放/平移参数，调制图像特征"),
        ("Self-Attention", "让当前批次中的 spots 互相传递信息"),
        ("表达解码", "预测 Harmony-PCA 50 维，再逆投影至基因"),
    ]
    for i, (hd, bd) in enumerate(steps):
        add_method_step(s, i + 1, hd, bd, 8.82, 1.10 + i * .91, 3.85, GREEN)
    box(s, .60, 5.95, 12.04, .70, PINK, PINK, True, 0)
    rich(s, [("需要注意：", True, RED), ("论文图中 self-attention 是在 N 个 spots 之间运算；官方代码实际把“当前 DataLoader batch”当作 N，因此邻域定义受 batching 影响。", False, INK)], .82, 6.16, 11.62, .30, 11.6)
    footer(s, "图源：Liu et al., Genome Biology (2026), Fig. 9, doi:10.1186/s13059-026-04168-x。")

    # 8 ResSAT 数据与复核
    s = blank(prs); title(s, "ResSAT 原始数据与本项目复核：作者协议不能与统一 benchmark 混用", 8, "02 候选论文")
    card(s, "主数据 SA：小鼠脑矢状前部", "Section 1：31,053 genes × 2,825 spots\nSection 2：31,053 genes × 2,696 spots\n两张切片轮换训练/测试", .62, 1.15, 3.90, 1.72, BLUE, NAVY, 11.5, 13.5)
    card(s, "主数据 SP：小鼠脑矢状后部", "Section 1：32,285 genes × 3,355 spots\nSection 2：32,285 genes × 3,289 spots\n同样采用两切片互换", 4.72, 1.15, 3.90, 1.72, GREEN, "40765A", 11.5, 13.5)
    card(s, "外部数据", "鳞状细胞癌：从 12 张中选 3 张\nHER2ST：从同一患者 36 张中选 6 张\n用于跨组织验证", 8.82, 1.15, 3.86, 1.72, YELLOW, GOLD, 11.5, 13.5)
    box(s, .62, 3.18, 5.85, 2.70, LIGHT, LINE, True, .8)
    text(s, "作者预处理协议", .86, 3.42, 2.35, .32, 15, RED, True)
    text(s, "1. 每个 spot 归一化到 10,000 后 log1p\n2. 选择 2,000 个高变基因（HVG）\n3. 标准化 → PCA 50 维\n4. 用 section identity 做 Harmony 校正\n5. 模型预测校正后的低维表示，再逆投影", .88, 3.88, 5.22, 1.65, 12, INK)
    box(s, 6.72, 3.18, 5.96, 2.70, PINK, PINK, True, .8)
    text(s, "本项目的两层复核", 6.98, 3.42, 2.45, .32, 15, RED, True)
    rich(s, [("作者示例：", True, INK), ("全基因 macro PCC = 0.651；Top-50 高表达基因 PCC = 0.880。", False, INK)], 6.98, 3.91, 5.24, .48, 11.7)
    rich(s, [("统一 benchmark：", True, INK), ("固定人肝 200 genes、固定切片划分后 macro PCC = 0.100。", False, INK)], 6.98, 4.52, 5.24, .48, 11.7)
    text(s, "结论：前者证明官方代码路径可跑通；后者才用于方法间公平比较。", 6.98, 5.19, 5.18, .42, 11.5, RED, True)
    footer(s, "ResSAT 原论文报告 SA/SP 平均 PCC 约 0.658/0.698；本项目作者示例和统一协议的基因、组织、切片划分均不同。")

    # 9 Stem
    s = blank(prs); title(s, "Stem：用条件扩散表达“一张图可能对应多种合理表达”", 9, "02 候选论文")
    box(s, .55, 1.08, 7.25, 4.87, WHITE, LINE, True, .8)
    add_picture_contain(s, ASSET / "stem_arch.png", .68, 1.21, 7.00, 4.58)
    add_method_step(s, 1, "条件信息", "UNI + CONCH 病理基础模型提取 H&E 特征；加入基因/类型 embedding", 8.02, 1.12, 4.64, PURPLE)
    add_method_step(s, 2, "训练", "向真实表达逐步加噪；DiT 学习在图像条件下预测噪声", 8.02, 2.03, 4.64, PURPLE)
    add_method_step(s, 3, "推断", "从高斯噪声反向去噪；同一 spot 可采样多个表达向量", 8.02, 2.94, 4.64, PURPLE)
    add_method_step(s, 4, "评价重点", "除 PCC 外强调 RVD：预测是否保留真实基因变异水平", 8.02, 3.85, 4.64, PURPLE)
    box(s, 8.02, 4.82, 4.64, 1.08, PINK, PINK, True, 0)
    text(s, "优势：能表示不确定性和一对多。\n代价：1,000 步扩散训练/采样慢，显存开销大。", 8.25, 5.04, 4.18, .65, 11.5, INK, True)
    footer(s, "图源：Zhu et al., ICLR 2025；Stem = SpaTially resolved gene Expression inference with diffusion Model。")

    # 10 Stem 数据
    s = blank(prs); title(s, "Stem 的四个原始数据集：跨器官、跨平台，但各自只留一张切片测试", 10, "02 候选论文")
    data = [
        ["数据集", "组织 / 平台", "规模", "目标基因", "测试切片"],
        ["Kidney", "人肾 / 10x Visium", "23 sections / 22 individuals\n315–4,159 spots", "200 HMHVG + 200 HVG", "20-0038 AKI"],
        ["HER2ST", "HER2+ 乳腺癌 / ST", "36 slices / 8 patients\n176–712 spots", "300 HMHVG + 296 DEGs", "B1"],
        ["PRAD", "前列腺癌 / Visium", "23 samples / 2 patients\n1,418–4,079 spots", "200 HMHVG", "MEND145"],
        ["Mouse Brain", "成年小鼠脑 / Visium", "14 samples / 4 mice\n2,675–3,617 spots", "200 HMHVG", "NCBI667"],
    ]
    add_table(s, data, .62, 1.20, 12.05, 3.75, [1.50, 2.30, 2.43, 2.35, 2.40], font_size=10)
    card(s, "训练细节", "224×224 patch；条件 DiT；训练使用 H&E 数据增强；论文默认 1:4 原图/增强图；扩散过程 1,000 steps。", .65, 5.25, 3.75, 1.22, LIGHT, RED, 11.2, 13)
    card(s, "当前工程适配", "官方脚本硬编码 UNI+CONCH、cuda:6 与分布式训练。本项目改为可用图像特征、单卡 AMP，并完成训练与 3 次/spot 采样。", 4.60, 5.25, 4.10, 1.22, LIGHT, NAVY, 11.2, 13)
    card(s, "公平比较风险", "论文数据、目标基因集合和测试切片均与 GSE240429 不同；其论文 PCC 不应直接抄到统一结果表。", 8.90, 5.25, 3.75, 1.22, LIGHT, GOLD, 11.2, 13)
    footer(s, "HMHVG = highly and moderately highly variable genes；DEG = differentially expressed genes。数据规模来自 Stem ICLR 2025 附录。")

    # 11 GenAR
    s = blank(prs); title(s, "GenAR：把表达当作离散计数 token，按基因层级粗到细生成", 11, "02 候选论文")
    box(s, .55, 1.08, 7.62, 4.98, WHITE, LINE, True, .8)
    add_picture_contain(s, ASSET / "genar_arch.png", .68, 1.19, 7.36, 4.70)
    add_method_step(s, 1, "基因层级", "按训练集空间表达模式聚类：1→4→8→40→100→200", 8.42, 1.12, 4.23, PINK)
    add_method_step(s, 2, "条件融合", "UNI 图像特征 + 坐标编码 → histological embedding", 8.42, 2.03, 4.23, PINK)
    add_method_step(s, 3, "因果生成", "12 层 causal Transformer 条件于已生成粗尺度，预测下一尺度", 8.42, 2.94, 4.23, PINK)
    add_method_step(s, 4, "物理输出", "输出离散词表中的整数 raw counts，而非连续 log 表达", 8.42, 3.85, 4.23, PINK)
    box(s, 8.42, 4.82, 4.23, 1.18, YELLOW, YELLOW, True, 0)
    text(s, "目的：显式建模基因共表达与零值；相较扩散减少约 6.4× FLOPs（论文报告）。", 8.66, 5.07, 3.74, .65, 11.2, INK, True)
    footer(s, "图源：Ouyang et al., Medical Image Analysis 113 (2026) 104232, doi:10.1016/j.media.2026.104232。")

    # 12 GenAR 数据
    s = blank(prs); title(s, "GenAR 的原始数据与评价口径：重点检查 Top-k 基因选择", 12, "02 候选论文")
    data = [
        ["数据集", "组织 / 规模", "测试切片", "基因与输入"],
        ["HER2ST", "HER2+ breast；13,594 spots", "SPA148", "Top 200 HE+HV；224 patch；UNI"],
        ["PRAD", "Prostate；23 slides；1,418–4,079 spots", "MEND145", "Top 200 HE+HV；raw integer counts"],
        ["Kidney", "23 slides；315–4,159 spots", "NCBI697", "Top 200 HE+HV；raw integer counts"],
        ["Mouse Brain", "14 slides；2,675–3,617 spots", "NCBI667", "Top 200 HE+HV；raw integer counts"],
        ["ccRCC（附录）", "Kidney cancer；24 slides", "INT2", "同一 coarse-to-fine 生成流程"],
    ]
    add_table(s, data, .62, 1.18, 12.05, 3.65, [1.60, 3.35, 2.10, 4.95], font_size=9.8)
    card(s, "论文的 PCC-10 / 50 / 200", "按测试结果挑选相关性最高的前 k 个基因再平均。它能反映“最好预测的一组基因”，但不是固定面板上的总体性能。", .65, 5.10, 3.82, 1.35, PINK, RED, 11.2, 13)
    card(s, "本项目严格主指标", "先在训练切片确定固定 200 基因，再对测试切片所有 200 基因逐基因计算 PCC 后取平均；不看测试结果选基因。", 4.67, 5.10, 3.82, 1.35, GREEN, "40765A", 11.2, 13)
    card(s, "当前复核值", "同一 GenAR 输出：测试后 Top-10/50/200 PCC = 0.166/0.130/0.072；严格固定 200 基因 macro PCC = 0.005。", 8.69, 5.10, 3.95, 1.35, YELLOW, GOLD, 11.2, 13)
    footer(s, "HE+HV = high-expression + high-variance genes。论文式 Top-k 作为补充指标保留，但不用于主结论。")

    # 13 横向方法比较
    s = blank(prs); title(s, "五种方法的本质差别：预测对象、上下文和失真方式不同", 13, "03 方法比较")
    data = [
        ["方法", "输出对象", "利用空间上下文", "核心训练目标", "主要优点", "主要风险"],
        ["ST-Net", "连续 log 表达", "否", "逐 spot 回归损失", "简单、快速、可解释", "过度平滑"],
        ["BLEEP", "邻居表达均值", "间接（参考库）", "图像-表达对比损失", "无需直接解码高维基因", "受训练库覆盖限制"],
        ["ResSAT", "PCA/Harmony 表达", "是（batch attention）", "低维表示回归", "融合坐标与跨 spot 关系", "batch 不是稳定空间邻域"],
        ["Stem", "表达分布样本", "通过条件特征", "扩散噪声预测", "能表达一对多与不确定性", "训练/采样昂贵"],
        ["GenAR", "离散 raw counts", "坐标条件 + 基因层级", "下一尺度 token 预测", "基因依赖与零值更自然", "训练难、域偏移敏感"],
    ]
    add_table(s, data, .48, 1.17, 12.40, 4.70, [1.12, 1.72, 1.90, 2.10, 2.55, 3.01], font_size=9.1)
    box(s, .62, 6.08, 12.03, .65, PINK, PINK, True, 0)
    text(s, "最终比较应回答两个问题：①平均数值是否准确；②空间结构、零值和生物异质性是否被保留。只看 PCC 会遗漏第二个问题。", .86, 6.27, 11.55, .28, 11.7, RED, True, align=PP_ALIGN.CENTER)
    footer(s, "“生成式”不等于结果一定更好：Stem/GenAR 的优势要在充足训练、匹配的数据分布和合适的生成指标下体现。")

    # 14 数据集全景
    s = blank(prs); title(s, "原论文数据集全景：不同论文的数字不能直接横向比", 14, "04 数据集")
    data = [
        ["论文", "主要组织", "样本规模", "目标基因", "原论文划分特点"],
        ["ST-Net", "Breast cancer", "68 sections / 23 patients / 30,612 spots", "250", "leave-one-patient-out"],
        ["BLEEP", "Human liver", "4 sections / 9,269 spots", "3,467", "留一张切片测试"],
        ["ResSAT", "Mouse brain SA/SP", "各 2 sections；2,696–3,355 spots/section", "2,000 HVG", "两切片互换；PCA+Harmony"],
        ["Stem", "Kidney / Breast / Prostate / Brain", "14–36 sections，依数据集而异", "200–1,000", "各数据集固定一张 test slide"],
        ["GenAR", "Breast / Prostate / Kidney / Brain / ccRCC", "14–24+ slides，依数据集而异", "200 HE+HV", "各数据集固定一张 test slide"],
    ]
    add_table(s, data, .55, 1.17, 12.25, 4.55, [1.30, 2.10, 3.05, 1.55, 4.25], font_size=9.5)
    card(s, "为什么不能把论文 PCC 排成一列", "组织不同、平台不同、基因数不同、归一化不同、测试切片不同；有的还只汇报 Top-k 基因。数值看似同名，实际统计对象不同。", .65, 5.88, 5.88, .98, LIGHT, RED, 10.5, 12.3)
    card(s, "统一 benchmark 的作用", "固定数据、划分、基因和指标，只让“方法”变化；因此当前统一结果比跨论文抄表更可信。", 6.78, 5.88, 5.88, .98, LIGHT, NAVY, 10.5, 12.3)
    footer(s, "说明：表中规模用于理解论文实验设计，不代表这些数据已全部下载到本项目。")

    # 15 HEST
    s = blank(prs); title(s, "HEST-1k：下一阶段跨组织 benchmark 的主要数据来源", 15, "04 数据集")
    box(s, .58, 1.10, 6.45, 5.35, WHITE, LINE, True, .8)
    add_picture_contain(s, ROOT / "third_party" / "HEST" / "figures" / "fig1.jpeg", .70, 1.22, 6.20, 5.08)
    card(s, "论文版本（2024）", "1,229 个 ST profile + 配对 WSI\n153 个 cohort / 26 个器官 / 人与小鼠\n367 个癌症样本 / 25 种癌症\n2.1M expression–morphology pairs\n76M+ nuclei", 7.30, 1.12, 2.52, 2.62, BLUE, NAVY, 11.2, 13)
    card(s, "官方仓库当前版本", "v1.3.0 已更新到 1,276 个配对样本，并加入 Visium HD 与 Xenium。\n\n需记录版本号，避免“HEST-1k 样本数”前后不一致。", 10.04, 1.12, 2.62, 2.62, GREEN, "40765A", 11.2, 13)
    card(s, "为什么先不全量下载", "全量 WSI 与表达数据体量很大；当前先用 4 张人肝切片建立可靠流程，再从 HEST 选择 2–3 个器官做外部验证。", 7.30, 4.02, 5.36, 1.18, LIGHT, RED, 11.5, 13)
    card(s, "准备使用的 HEST 能力", "统一读取、WSI/spot 对齐、标准 224px patch、元数据筛选、按病人或 cohort 防泄漏划分。", 7.30, 5.40, 5.36, 1.03, LIGHT, GOLD, 11.2, 13)
    footer(s, "图源：Jaume et al., HEST-1k, NeurIPS 2024；官方仓库：https://github.com/mahmoodlab/hest（当前版本信息核对于 2026-09-01）。")

    # 16 GSE 数据收集
    s = blank(prs); title(s, "最终统一 benchmark：GSE240429 四张连续人肝切片", 16, "05 Benchmark")
    box(s, .55, 1.10, 6.35, 5.35, WHITE, LINE, True, .8)
    add_picture_contain(s, ASSET / "gse240429_four_slides.png", .68, 1.21, 6.10, 5.12)
    card(s, "数据来源与生物学背景", "NCBI GEO：GSE240429\n人类供体肝脏，10x Genomics Visium\n研究健康与 PSC（原发性硬化性胆管炎）免疫微环境\n4 张连续 16 μm 切片", 7.18, 1.12, 5.48, 1.78, BLUE, NAVY, 11.6, 13.5)
    card(s, "为什么选它做首个 benchmark", "• 与 BLEEP 原论文完全对应\n• 四张切片允许 train / val / test 分离\n• 同一组织降低早期变量数量\n• 每张约 2.3k spots，单卡可完成多方法比较", 7.18, 3.12, 5.48, 1.78, GREEN, "40765A", 11.6, 13.5)
    card(s, "本地收集状态", "GEO 原始包 6.4 GB 已下载；4 张完整 TIFF 均约 2.6 GB；表达矩阵、坐标、缩略图和全分辨率 H&E 均已配对。", 7.18, 5.12, 5.48, 1.22, PINK, RED, 11.5, 13.5)
    footer(s, "GEO 样本：GSM7697868–GSM7697871；总计 9,269 个组织内 spots。缩略图来自本项目本地原始数据。")

    # 17 预处理与划分
    s = blank(prs); title(s, "统一预处理与数据划分：所有方法必须遵守同一条流水线", 17, "05 Benchmark")
    stages = [
        ("原始数据", "TIFF + counts + spot 坐标", BLUE),
        ("表达处理", "每 spot 1e4\n+ log1p", GREEN),
        ("固定基因", "仅训练集选 200 genes", YELLOW),
        ("图像输入", "每 spot 裁 224×224 patch", PINK),
        ("缓存", "patch uint8 + ResNet18 512D", PURPLE),
        ("训练评测", "A1+B1 / C1 / D1", BLUE),
    ]
    for i, (hd, bd, fill) in enumerate(stages):
        x = .48 + i * 2.13
        box(s, x, 1.42, 1.82, 1.65, fill, LINE, True, .8)
        text(s, hd, x + .10, 1.65, 1.62, .30, 12.5, RED, True, align=PP_ALIGN.CENTER)
        text(s, bd, x + .15, 2.12, 1.52, .58, 10.5, INK, False, align=PP_ALIGN.CENTER, valign=MSO_ANCHOR.MIDDLE)
        if i < 5:
            arrow(s, x + 1.84, 2.24, x + 2.06, 2.24, GOLD, 1.5)
    card(s, "切片级划分", "训练：C73_A1 + C73_B1 = 4,727 spots\n验证：C73_C1 = 2,277 spots\n测试：C73_D1 = 2,265 spots", .65, 3.46, 3.75, 1.55, LIGHT, RED, 12, 13.5)
    card(s, "训练集内选择 200 基因", "检测率 ≥5%；去重；排除 MT、RPS/RPL；在训练集 8,485 个候选中按 log 表达方差选择。基因顺序用 SHA256 固定。", 4.62, 3.46, 4.05, 1.55, LIGHT, NAVY, 11.2, 13.5)
    card(s, "为什么不能随机拆 spots", "相邻 spot 形态和表达高度相关。随机拆分会让同一切片同时出现在训练与测试，产生严重空间泄漏；必须按完整切片划分。", 8.88, 3.46, 3.78, 1.55, LIGHT, GOLD, 11.2, 13.5)
    box(s, .65, 5.32, 12.01, 1.10, PINK, PINK, True, 0)
    text(s, "Cache 的位置与作用", .90, 5.54, 1.95, .28, 13.5, RED, True)
    text(s, "最耗时的 TIFF 解码、spot 裁图和 CNN 特征提取只做一次；后续每个 epoch 直接读 patch/512D 特征，避免重复处理约 10.5 GB 的全分辨率图像。", 2.65, 5.51, 9.64, .55, 11.8, INK)
    footer(s, "固定基因面板 SHA256：96aab61d…c57b9c7；该指纹可检查不同方法是否真的使用同一列顺序。")

    # 18 当前工程
    s = blank(prs); title(s, "当前工程进展：从真实数据到五类模型的统一结果已经跑通", 18, "06 当前进展")
    items = [
        ("01", "数据下载与完整性审计", "4 TIFF / counts / coordinates 全部校验", GREEN),
        ("02", "表达与图像预处理", "9,269 spots；200 genes；224px patch", GREEN),
        ("03", "缓存与基线", "ResNet18 特征；均值/坐标/图像回归", GREEN),
        ("04", "论文方法适配", "BLEEP / ResSAT / GenAR / Stem", GREEN),
        ("05", "统一评测", "8 项指标 + 空间热图 + 稀疏性审计", GREEN),
        ("06", "待扩展", "多随机种子、更多器官、标准病理 foundation encoder", YELLOW),
    ]
    for i, (num, hd, bd, fill) in enumerate(items):
        col, row = i % 3, i // 3
        x, y = .62 + col * 4.08, 1.25 + row * 2.18
        box(s, x, y, 3.78, 1.75, fill, LINE, True, .8)
        text(s, num, x + .18, y + .16, .55, .38, 18, RED, True)
        text(s, hd, x + .82, y + .18, 2.70, .32, 13.5, INK, True)
        text(s, bd, x + .82, y + .71, 2.65, .62, 11.2, INK)
        text(s, "已完成" if i < 5 else "下一阶段", x + .20, y + 1.35, 1.15, .23, 9.5, "40765A" if i < 5 else GOLD, True)
    box(s, .62, 5.75, 12.02, .72, PINK, PINK, True, 0)
    text(s, "现阶段结论是“统一流程已跑通并形成可审计结果”，不是“所有论文都已达到作者指标”。后者需要匹配原始数据、模型规模与训练预算。", .88, 5.96, 11.48, .32, 11.8, RED, True, align=PP_ALIGN.CENTER)
    footer(s, "主要产物：data/processed/gse240429、results/unified_comparison.json、figures/04–05、各模型 checkpoint 与训练日志。")

    # 19 指标
    s = blank(prs); title(s, "评价指标：主指标固定所有基因，同时检查误差、结构与生成特性", 19, "07 评价协议")
    metric_cards = [
        ("Macro PCC", "对每个基因跨测试 spots 计算 Pearson，再对固定 200 基因平均。主排序指标。", BLUE, NAVY),
        ("Macro Spearman", "只比较排序关系，对极端值更稳健；同样对 200 基因平均。", GREEN, "40765A"),
        ("MAE / RMSE", "衡量 log-normalized 表达的绝对误差；RMSE 更惩罚大误差。", YELLOW, GOLD),
        ("Spot cosine", "逐 spot 比较整条 200 基因表达向量方向；高均值可能被高表达基因主导。", PURPLE, "6B4D8A"),
        ("RVD", "Relative Variation Distance：预测与真值的基因变异程度是否接近；越小越好。", PINK, RED),
        ("Moran's I", "衡量空间自相关；比较每个基因的空间结构，而不只比较数值。", BLUE, NAVY),
    ]
    for i, (hd, bd, fill, accent) in enumerate(metric_cards):
        col, row = i % 3, i // 3
        card(s, hd, bd, .62 + col * 4.08, 1.18 + row * 2.05, 3.77, 1.62, fill, accent, 11.2, 13.5)
    box(s, .62, 5.48, 12.01, 1.00, PINK, PINK, True, 0)
    rich(s, [("不作为主指标：", True, RED), ("PCC-10/50/200（先看测试结果再挑 Top-k 基因）和作者各自协议下的单一分数。它们只作为论文复核或补充说明。", False, INK)], .90, 5.76, 11.42, .46, 11.6)
    footer(s, "所有统一指标均在 held-out C73_D1 上重算；预测与真值使用同一固定 200 基因顺序。")

    # 20 量化结果
    s = blank(prs); title(s, "统一 benchmark 当前结果：ResSAT 相关性最高，但绝对水平仍低", 20, "08 实验结果")
    chart_data = CategoryChartData()
    chart_data.categories = ["坐标岭回归", "坐标KNN", "图像岭回归", "ST-Net式MLP", "BLEEP", "ResSAT", "GenAR", "Stem"]
    chart_data.add_series("Macro PCC", [-.0047, -.0024, .0589, .0571, .0515, .0998, .0051, .0025])
    chart_data.add_series("Macro Spearman", [.0052, .0048, .0469, .0441, .0481, .0730, -.0039, .0029])
    chart = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(.55), Inches(1.16), Inches(8.05), Inches(4.65), chart_data).chart
    chart.has_legend = True; chart.legend.position = XL_LEGEND_POSITION.BOTTOM; chart.legend.font.size = Pt(9)
    chart.has_title = False
    chart.value_axis.minimum_scale = -.01; chart.value_axis.maximum_scale = .11; chart.value_axis.major_unit = .02
    chart.value_axis.tick_labels.font.size = Pt(9); chart.category_axis.tick_labels.font.size = Pt(8)
    chart.series[0].format.fill.solid(); chart.series[0].format.fill.fore_color.rgb = rgb(NAVY)
    chart.series[1].format.fill.solid(); chart.series[1].format.fill.fore_color.rgb = rgb(SKY)
    card(s, "严格协议下的最好结果", "ResSAT\nMacro PCC 0.100\nSpearman 0.073\nMAE 0.579", 8.87, 1.20, 3.78, 1.52, GREEN, "40765A", 12, 13.5)
    card(s, "必须保留的负面结论", "训练均值基线 MAE=0.582，ResSAT 仅改善约 0.002；所有方法 macro PCC ≤0.10，尚不能声称达到论文水平。", 8.87, 2.94, 3.78, 1.55, PINK, RED, 11.5, 13.5)
    card(s, "指标之间会冲突", "Stem 的 Moran-I MAE 最低（0.045），但 PCC≈0；GenAR spot cosine=0.847 较高，却没有恢复逐基因变化。", 8.87, 4.70, 3.78, 1.40, YELLOW, GOLD, 11.2, 13)
    footer(s, "测试集：C73_D1，2,265 spots × 固定 200 genes。结果文件：results/unified_comparison.json。")

    # 21 定性结果
    s = blank(prs); title(s, "空间热图检查：单个代表基因可以较好，但不能替代 200 基因总体指标", 21, "08 实验结果")
    box(s, .54, 1.07, 8.30, 5.70, WHITE, LINE, True, .8)
    add_picture_contain(s, ASSET / "qualitative_selected.png", .67, 1.18, 8.03, 5.47)
    card(s, "三列代表什么", "GLUL、CYP1A2：肝细胞分区相关基因\nIGKC：免疫相关且更稀疏\n第一行是真值，后续为不同方法预测。", 9.08, 1.12, 3.55, 1.52, LIGHT, NAVY, 11.5, 13)
    card(s, "当前观察", "ResSAT 在这三个基因上 PCC=0.495 / 0.567 / 0.243，能恢复部分大尺度分区；GenAR 和 Stem 当前适配版本更偏平滑或失真。", 9.08, 2.88, 3.55, 1.70, GREEN, "40765A", 11.3, 13)
    card(s, "为什么不能只放好看的热图", "代表基因是人为挑选的；总体结论仍必须来自预先固定的 200 基因 macro 指标，并报告失败基因和稀疏性。", 9.08, 4.82, 3.55, 1.55, PINK, RED, 11.3, 13)
    footer(s, "热图均来自 held-out C73_D1，颜色为统一 log1p(1e4) 表达尺度；完整图见 figures/05_gene_spatial_predictions.png。")

    # 22 审计发现
    s = blank(prs); title(s, "复现审计发现：论文代码可运行不等于实验定义已经对齐", 22, "09 独立思考")
    findings = [
        ("01", "数据包完整性", "GEO 6.4 GB tar 中 D1 TIFF 曾截断，缺少约 343 MB；已单独重下并用 gzip ISIZE 验证。"),
        ("02", "ResSAT 输入类型", "官方示例 patch 为 float32 的 0–255；新版 torchvision 的 ToPILImage 会再次缩放并取模，需转 uint8。"),
        ("03", "ResSAT spot 交互", "self-attention 作用于任意 batch，而非固定物理邻域。当前权重近似单位映射，batch 变化影响虽小但定义仍不严谨。"),
        ("04", "GenAR 协议漂移", "论文附录、仓库配置与当前环境默认值存在差异；层级、训练 epoch 与评测 Top-k 必须显式记录。"),
        ("05", "Stem 工程假设", "官方脚本默认 UNI+CONCH、cuda:6/DDP；8 GB GPU 下 fp32 不可用，AMP batch=32 后约 0.07 s/step、峰值约 2 GB。"),
        ("06", "稀疏性恢复不足", "测试真值零比例 24.7%；GenAR 预测零比例仅 3.7%、zero-F1=0.112；连续回归则几乎没有精确零值。"),
    ]
    for i, (num, hd, bd) in enumerate(findings):
        col, row = i % 2, i // 2
        x, y = .60 + col * 6.08, 1.15 + row * 1.72
        box(s, x, y, 5.82, 1.42, LIGHT if row % 2 == 0 else WHITE, LINE, True, .8)
        text(s, num, x + .16, y + .16, .48, .30, 15, RED, True)
        text(s, hd, x + .74, y + .14, 2.10, .28, 12.3, INK, True)
        text(s, bd, x + .74, y + .52, 4.76, .65, 10.6, INK)
    box(s, .60, 6.43, 11.90, .35, PINK, PINK, True, 0)
    text(s, "这些问题均已记录到日志/结果文件；它们不是“调参细节”，而是决定结果能否被解释和复查的实验定义。", .82, 6.50, 11.45, .19, 10.7, RED, True, align=PP_ALIGN.CENTER)
    footer(s, "审计证据位于 data/、results/logs_*、docs/03_FINAL_REPORT.md；所有修改均保留作者原仓库作为 third_party 对照。")

    # 23 下一步
    s = blank(prs); title(s, "下一步：先提高比较可信度，再追求更高分数", 23, "10 研究计划")
    roadmap = [
        ("近期：稳健性", "每个方法至少 3 个随机种子；报告均值±标准差；统一早停与训练预算。", GREEN),
        ("近期：编码器", "将共享 ResNet18 替换为同一病理 foundation encoder，分离“模型结构”和“图像特征质量”。", BLUE),
        ("中期：空间建模", "把 ResSAT 的 batch attention 改成基于坐标的 kNN/半径邻域；跨 batch 保持同一空间图。", YELLOW),
        ("中期：生成评测", "增加零值 precision/recall、分位数误差、校准曲线和多次采样不确定性。", PURPLE),
        ("中期：外部验证", "从 HEST 选择肾、乳腺、前列腺等 2–3 个器官；按患者/cohort 拆分，检查域泛化。", PINK),
        ("长期：统一模型", "考虑多组织条件模型或 foundation model；输出完整转录组，并服务下游聚类、分区和细胞类型分析。", GREEN),
    ]
    for i, (hd, bd, fill) in enumerate(roadmap):
        col, row = i % 2, i // 2
        card(s, hd, bd, .62 + col * 6.05, 1.16 + row * 1.72, 5.78, 1.40, fill, RED if col == 0 else NAVY, 11.2, 13.2)
    box(s, .62, 6.40, 11.83, .38, PINK, PINK, True, 0)
    text(s, "目标：形成一套可复用的 H&E→ST benchmark，而不是只复现某一篇论文的单一分数。", .88, 6.48, 11.30, .20, 11.2, RED, True, align=PP_ALIGN.CENTER)
    footer(s, "优先级建议：多种子复测 → 统一病理编码器 → 空间邻域修正 → HEST 跨组织验证。")

    # 24 结束页
    s = blank(prs)
    s.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.RECTANGLE, 0, 0, Inches(.18), H).fill.solid()
    s.shapes[-1].fill.fore_color.rgb = rgb(RED); s.shapes[-1].line.fill.background()
    text(s, "汇报总结", .72, .72, 2.0, .32, 12, RED, True)
    text(s, "已完成真实数据、五类模型与统一评价的\n端到端实验闭环", .72, 1.43, 7.35, 1.15, 27, INK, True)
    card(s, "已经回答", "任务和数据是什么；各方法怎样工作；原论文数据如何设置；统一 benchmark 怎样收集、划分和评价。", .76, 3.05, 3.65, 1.42, LIGHT, RED, 11.5, 13)
    card(s, "当前最可信结论", "ResSAT 在固定 200 基因上相关性最高，但绝对成绩仍低；现阶段重点应是稳健复测和跨组织泛化。", 4.65, 3.05, 3.65, 1.42, LIGHT, NAVY, 11.5, 13)
    card(s, "请老师指导", "候选方法取舍、benchmark 扩展顺序，以及更适合评价生成质量的生物学指标。", 8.54, 3.05, 3.65, 1.42, LIGHT, GOLD, 11.5, 13)
    text(s, "请老师批评指正", .72, 5.70, 11.48, .52, 22, RED, True, align=PP_ALIGN.CENTER)
    text(s, "谢谢！", .72, 6.35, 11.48, .34, 13, MUTED, False, align=PP_ALIGN.CENTER)

    # 清理母版占位符并写入元数据。
    prs.core_properties.title = "基于 H&E 染色图像的空间转录组数据生成研究"
    prs.core_properties.subject = "候选论文、原始数据集、统一 benchmark 与当前实验进展"
    prs.core_properties.author = "庹力元"
    prs.core_properties.comments = "由项目真实数据、论文原图与本地实验结果生成；2026-09-01。"
    prs.save(OUT)
    print("saved presentation")


if __name__ == "__main__":
    build_deck()
