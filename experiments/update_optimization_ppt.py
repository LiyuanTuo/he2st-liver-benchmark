"""Refresh pages 9/11 of the existing 11-page deck; preserve original evidence."""
from pathlib import Path
import importlib.util
import json
import shutil

from pptx import Presentation

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/optimization_20260914'
spec = importlib.util.spec_from_file_location('style', ROOT/'scripts/53_update_verified_ppt.py')
style = importlib.util.module_from_spec(spec); spec.loader.exec_module(style)


def main():
    selection = json.loads((OUT/'selection_locked.json').read_text())
    test = json.loads((OUT/'test_metrics.json').read_text())['test']
    name = selection['selected']; best = test[name]
    backup = OUT/'ppt_backup'; backup.mkdir(exist_ok=True)
    for suffix in ['.pptx', '.pdf']:
        path = style.PPT.with_suffix(suffix)
        if path.exists() and not (backup/path.name).exists():
            shutil.copy2(path, backup/path.name)
    prs = Presentation(style.PPT); assert len(prs.slides)==11
    # Page 6 and maps remain a clearly labelled previous-round comparison.
    s = prs.slides[5]
    for shape in s.shapes:
        if shape.has_text_frame:
            for p in shape.text_frame.paragraphs:
                for run in p.runs:
                    run.text = run.text.replace('统一人肝 Benchmark：七种基线与自建模型', '前轮基准：七种基线与自建模型')
                    run.text = run.text.replace('同一 D1、训练选定 HEG200 / HEG50、同一未重建真值；新配置只按 C1 选择。', '2026-09-13 已验证结果；本轮新增优化对照与最终选择见第9页。')
    txt, card = style.txt, style.card
    s = prs.slides[8]
    style.header(s,9,'新一轮优化：固定面板、验证选择、测试实测',
                 '保持 A1+B1 / C1 / D1 与 HEG200 / HEG50；不改测试真值。三组新训练与全部混合候选均保留。',
                 '来源：benchmarks/liver/optimization_20260914.csv；docs/RESULTS.md。EMA是候选权重，不保证被选中。',
                 json.dumps({'selection':selection,'test':test}, ensure_ascii=False)+'\n本批按C1选方案后才评价D1，但D1在整个项目历史中被反复查看，仍需独立供体外部测试。')
    labels = {'resnet18_ema':'ResNet18 / 新训练配方', 'efficientnet_b0_ema':'EfficientNet-B0 / 新配方',
              'resnet18_spatial':'ResNet18 / 训练标签平滑', 'new_equal_ensemble':'三个新模型等权',
              'previous_contextfusion':'上一轮多种子集成', 'new_weight_0.25':'新模型权重 25%',
              'new_weight_0.5':'新模型权重 50%', 'new_weight_0.75':'新模型权重 75%'}
    cols = [(.68,3.8),(4.52,1.95),(6.57,1.95),(8.63,1.9),(10.66,1.7)]
    for (x,w), label in zip(cols, ['候选方案','C1 HEG200','D1 HEG200','D1 HEG50','D1 MAE']):
        txt(s,label,x,1.92,w,.37,14,style.RED,True)
    for i,(key,m) in enumerate(test.items()):
        y=2.40+i*.36
        if key==name:
            style.rect(s,.60,y-.02,12.1,.35,'EDF5EF')
        values=[labels.get(key,key)+(' ✓' if key==name else ''),f"{selection['validation'][key]['HEG200']:.4f}",f"{m['HEG200']:.4f}",f"{m['HEG50']:.4f}",f"{m['MAE']:.3f}"]
        for (x,w),value in zip(cols,values):
            txt(s,value,x,y,w,.32,13,style.INK,key==name)
    txt(s,'共同配方：warmup + 余弦学习率、EMA 候选、颜色增强；平滑组只修改训练监督。',.70,5.48,12,.40,14,style.GRAY)
    previous = test.get('previous_contextfusion',best)
    delta=best['HEG200']-previous['HEG200']
    txt(s,f"C1 最终选择的 D1：{best['HEG200']:.4f} / {best['HEG50']:.4f}；HEG200 较前轮 {delta:+.4f}。",.70,6.02,12,.48,19,style.RED,True)
    txt(s,'仅小幅改善：HEG50增益区间跨零，MAE略变差；固定面板平均仍未达到0.5。',.70,6.54,12,.32,14,style.INK)
    s=prs.slides[10]
    style.header(s,11,'可复现交付与下一步研究',
                 '本轮完成三组真实训练、固定测试评价与仓库整理；保留全部候选及未改善的结果。',
                 '代码入口：run.py；运行指南：docs/QUICKSTART.md；结果与局限：docs/RESULTS.md。',
                 '仓库未推送GitHub；数据、权重、论文、PPT由.gitignore排除，仍留在本地。预测文件不含测试RNA。单供体且D1反复用于开发，不代表外部独立验证。')
    card(s,.60,1.93,5.95,2.15,'① 统一命令行与清晰代码结构',
         'run.py：prepare / fit / predict / evaluate\nsrc/he2st：数据、网络、训练和计分\nconfigs：实验参数；benchmarks：固定面板与结果\n历史脚本保留原路径与索引。','EDF3F8',15)
    card(s,6.77,1.93,5.95,2.15,'② 数据与预测边界已实测',
         '四张切片重建数组逐元素一致。\n144个跨尺度裁图抽查完全一致。\npredict可以使用完全没有RNA的输入。\n权重、C1选择记录和全部候选预测均保存。','EDF5EF',15)
    card(s,.60,4.34,5.95,2.15,'③ 当前结论与成绩',
         f"固定 D1 PCC：{best['HEG200']:.4f} / {best['HEG50']:.4f}\n对应固定 HEG200 / HEG50，均非测试挑基因。\n原ST-Net为0.1472 / 0.1978。\n仍不能声称固定面板PCC达到0.5。",'F9EDEC',15)
    card(s,6.77,4.34,5.95,2.15,'④ 后续优先补足泛化证据',
         '增加独立供体和切片，设未查看的外部测试。\n检验染色/采集差异及空间变化强度。\n保持高表达面板，报告逐基因误差与负结果。\n原论文鼠脑重建目标与人肝真值分开讨论。','F7F4EF',15)
    prs.save(style.PPT)
    print('Updated original deck:',style.PPT,'slides=',len(prs.slides))


if __name__ == '__main__':
    main()
