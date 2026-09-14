"""Point project entry pages at the completed follow-up, keeping historical results."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/liver_context_20260913'
def main():
    r=json.loads((OUT/'comparison.json').read_text());name=r['selected_by_C1'];best=r['metrics'][name];scratch=r['metrics']['ContextFusion-scratch']
    path=ROOT/'README.md';old=path.read_text(encoding='utf8');marker='> **2026-09-08 历史更新**';assert marker in old
    top=f'''# 基于 H&E 的空间转录组生成

> **2026-09-13 续研已完成**：在原来相同的 **HEG200 / HEG50、A1+B1/C1/D1、未重建真值** 上，新增三尺度端到端模型、全随机训练、单视野消融及三个预训练种子。
> C1 选定 `{name}`：D1 PCC **{best['HEG200']:.4f} / {best['HEG50']:.4f}**；原 ST-Net 为 **0.1472 / 0.1978**。全随机从零训练版本为 **{scratch['HEG200']:.4f} / {scratch['HEG50']:.4f}**。固定面板平均是否达0.5以实测值为准，不用个别高分基因代替。
> [原因、计数降采样与完整消融](docs/06_LIVER_CONTEXT_IMPROVEMENT.md) · [统一结果CSV](results/liver_context_20260913/combined_benchmark.csv) · [逐基因PCC](results/liver_context_20260913/combined_per_gene.csv)。
> [预测AnnData（含H&E及坐标，不含测试RNA）](results/liver_context_20260913/C73_D1_HE_predicted_HEG200.h5ad) · [预测log表达CSV](results/liver_context_20260913/D1_predicted_panel_log1p.csv.gz)。
> 原[统一Benchmark PPT](H&E空间转录组生成研究_统一Benchmark汇报_庹力元.pptx)就地更新为 **11页**，[PDF](H&E空间转录组生成研究_统一Benchmark汇报_庹力元.pdf)同步。权重、选择记录、失败控制、空间bootstrap与重推理核验均保存在 `results/liver_context*`。
> 上轮 [ResSAT复现与历史纠错](docs/05_PCC_DIAGNOSIS.md)仍保留：SA/SP论文口径HEG50为0.8803/0.8930；该鼠脑重建目标与当前人肝任务不能混比。
> 下方为历史进度；当前方法和成绩以上述续研报告为准。

'''
    path.write_text(top+old[old.index(marker):],encoding='utf8')
    path=ROOT/'docs/05_PCC_DIAGNOSIS.md';old=path.read_text(encoding='utf8')
    notice='> 后续已完成自建模型、全随机训练、多种子与计数诊断。最新结果见 [人肝续研报告](06_LIVER_CONTEXT_IMPROVEMENT.md)；本页保留上一轮原始复现与纠错证据。\n\n'
    if notice not in old:path.write_text(notice+old,encoding='utf8')
    status='''# 续研已完成（2026-09-13）

57–71脚本对应的实验、独立重推理、计数诊断、统一报告与11页PPT/PDF均已完成。
入口：docs/06_LIVER_CONTEXT_IMPROVEMENT.md；当前指标 comparison.json、combined_benchmark.csv；验证 final_deliverable_audit.json。
原始七方法结果和先前论文复现保留在 results/verified_20260913/，作为历史证据而非最新完整表。
'''
    (OUT/'STATUS.md').write_text(status,encoding='utf8');print(name,best)
if __name__=='__main__':main()
