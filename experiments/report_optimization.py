"""Publish small, traceable result snapshots; leave predictions and weights local."""
from pathlib import Path
import hashlib
import json
import shutil
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from he2st.metrics import per_gene_pearson


def main():
    output = Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'results/optimization_20260914'
    public = ROOT/'benchmarks/liver'
    assets = ROOT/'docs/assets'; assets.mkdir(exist_ok=True)
    select = json.loads((output/'selection_locked.json').read_text())
    result = json.loads((output/'test_metrics.json').read_text())
    rows = []
    for name, metric in result['test'].items():
        v = select['validation'][name]
        rows.append({'candidate': name, 'selected_on_C1': name==select['selected'],
                     'C1_HEG200': v['HEG200'], 'C1_HEG50': v['HEG50'],
                     'D1_HEG200': metric['HEG200'], 'D1_HEG50': metric['HEG50'], 'D1_MAE': metric['MAE']})
    frame = pd.DataFrame(rows)
    frame.to_csv(public/'optimization_20260914.csv', index=False)
    shutil.copy2(ROOT/'results/liver_context_20260913/combined_benchmark.csv', public/'baseline_20260913.csv')
    for name in ['plan.json', 'selection_locked.json', 'test_metrics.json']:
        shutil.copy2(output/name, public/f'optimization_{name}')
    with np.load(output/'test_candidates.npz') as d:
        per_gene = pd.DataFrame({'gene': d['genes'], 'HEG50': d['heg50_mask']})
        for name in result['test']:
            per_gene[name] = per_gene_pearson(d['truth'], d[name])
        per_gene.to_csv(public/'optimization_per_gene.csv', index=False)
        # The prediction deliverable contains no measured RNA.
        np.savez_compressed(output/'prediction_only.npz', predicted=d[select['selected']],
                            genes=d['genes'], barcodes=d['barcodes'], coordinates_xy=d['coordinates_xy'], heg50_mask=d['heg50_mask'])
        pd.DataFrame(d[select['selected']], index=d['barcodes'], columns=d['genes']).to_csv(output/'predicted_log_expression.csv.gz', index_label='barcode')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    names = ['resnet18_ema', 'efficientnet_b0_ema', 'resnet18_spatial']
    fitting = {}
    for name in names:
        history = json.loads((output/name/'history.json').read_text())
        selected = json.loads((output/name/'selection.json').read_text())
        fitting[name] = {'selection': selected, 'epochs_run': len(history), 'training_seconds': history[-1]['elapsed_seconds']}
        axes[0].plot([h['epoch'] for h in history], [max(h['validation'][k]['HEG200'] for k in ['raw','ema']) for h in history], label=name)
    axes[0].set(xlabel='Epoch', ylabel='C1 HEG200 PCC', title='Validation only: raw / EMA max')
    axes[0].legend(fontsize=8)
    compact = frame[frame.candidate.isin(names+['previous_contextfusion',select['selected']])]
    labels = compact.candidate.str.replace('previous_contextfusion','previous ensemble').str.replace('efficientnet_b0_ema','EfficientNet-B0')
    x = np.arange(len(compact)); width=.38
    axes[1].bar(x-width/2, compact.D1_HEG200, width, label='D1 HEG200')
    axes[1].bar(x+width/2, compact.D1_HEG50, width, label='D1 HEG50')
    axes[1].set_xticks(x, labels, rotation=25, ha='right', fontsize=8)
    axes[1].set(ylabel='PCC', title='Fixed test slide; all fixed genes', ylim=(0,.55))
    axes[1].axhline(.5, color='grey', ls='--', lw=.8); axes[1].legend(fontsize=8)
    fig.savefig(assets/'optimization_20260914.png', dpi=160); plt.close(fig)
    (public/'optimization_fits.json').write_text(json.dumps(fitting, indent=2), encoding='utf8')
    best = result['test'][select['selected']]
    previous = result['test'].get('previous_contextfusion')
    delta = '' if previous is None else f'相对上一轮 C1 选定集成，HEG200 变化 **{best["HEG200"]-previous["HEG200"]:+.4f}**，HEG50 变化 **{best["HEG50"]-previous["HEG50"]:+.4f}**。'
    uncertainty = ''
    if (output/'verification.json').exists():
        verified = json.loads((output/'verification.json').read_text())
        a = verified['CI95']['delta_HEG200_vs_previous']
        b = verified['CI95']['delta_HEG50_vs_previous']
        uncertainty = f'\n\n**增益幅度有限。** 24 个空间块、300 次配对 bootstrap 的增益 95% 区间：HEG200 [{a[0]:.4f}, {a[1]:.4f}]；HEG50 [{b[0]:.4f}, {b[1]:.4f}]，后者跨过零。该区间仅描述本张切片，不能代表独立患者或消除反复开发的偏倚。MAE 从 0.4256 增至 0.4274，误差指标略有变差。三组新权重重新加载推理后与保存预测逐元素一致。见 [核验与区间](../benchmarks/liver/optimization_verification.json)。'
    table = '\n'.join('| '+ ' | '.join([r['candidate']+(' **← C1选择**' if r['selected_on_C1'] else ''), *[f'{r[k]:.4f}' for k in ['C1_HEG200','C1_HEG50','D1_HEG200','D1_HEG50','D1_MAE']]])+' |' for r in rows)
    report = f'''# 本轮优化与仓库整理（2026-09-14）

本轮按 C1 选择 **{select['selected']}**，在固定 D1 上 PCC 为 **{best['HEG200']:.4f} / {best['HEG50']:.4f}（HEG200 / HEG50）**。{delta}固定面板平均仍未达到 0.5，不能以验证集、测试后挑选基因或平滑真值替代。
{uncertainty}

## 本次实际完成的对照

保持 A1+B1 / C1 / D1、固定高表达基因面板和未重建真值。三组方案均从 ImageNet 初始化，在训练集端到端训练：ResNet18、EfficientNet-B0，以及训练标签 20% 邻域平滑的 ResNet18。共同使用 warmup + 余弦学习率、EMA、颜色增强。每组在 C1 选原权重/EMA和单次/四旋转推理；再选单模型、三模型均值及预先声明的旧集成混合比例。

这是三组训练方案的比较；共同改变了多个训练因素，不能把差异单独归因于 EMA。`spatial` 与 `resnet18_ema` 只差训练标签平滑，可以直接作该因素的对照。

本轮三组最终均选中了原权重，EMA 没有被选中。新单模型均未超过上一轮集成。训练标签平滑使 D1 HEG200 从 0.2182 降到 0.2010，HEG50 从 0.2947 降到 0.2635，当前证据不支持继续加大平滑强度。最终小幅增益来自新旧预测组合，而非单个模型的明显突破。

| 候选 | C1 HEG200 | C1 HEG50 | D1 HEG200 | D1 HEG50 | D1 MAE |
|---|---:|---:|---:|---:|---:|
{table}

模型行名含 `ema` 表示训练过程中维护 EMA，并不表示最终一定选择 EMA；各组最终权重类型、轮次、TTA 与耗时见 [训练记录](../benchmarks/liver/optimization_fits.json)。混合候选中 `new_weight` 是 C1 最优新候选的权重，其余为上一轮集成；最优新候选是 `{select['best_new']}`。

![验证曲线与固定测试表现](assets/optimization_20260914.png)

## 为什么仍有差距

1. 历史错误已经修复：当前面板确实是训练集高表达基因；新模型训练与计分使用相同面板归一化，测试真值没有重建。本轮从源 counts 重建四张切片，五类数组字段与原数据逐元素一致；144 个跨切片、跨尺度裁图抽查完全一致。
2. 前轮模型 HEG50 从训练 0.608 降到 C1 0.485、D1 0.302，存在明显泛化差距。更多训练轮数与更复杂编码器是否有用，要看上表，不能只看训练损失。原 ST-Net 的 D1 为 0.1472 / 0.1978，前轮改进已证明能学习部分形态相关信号，但不能推导固定面板应达到 0.5。
3. C1 与 D1 的捕获 UMI 中位数为 12423 与 6010。前轮固定预测、只对 C1 做计数降采样时，HEG200 从 0.3485 降至约 0.3033。这支持计数差异有贡献，但不能解释全部差距，也不是性能硬上限。高平均表达不保证强空间变化。
4. 单供体仅两张训练切片，且 D1 已在多轮研究中被查看；这是反复使用的开发基准，仍需新的独立供体做最终外部测试。ResSAT 鼠脑与重建目标的论文数值不能直接作为这个任务的应达阈值。

## 可复查产物

- [全部候选指标](../benchmarks/liver/optimization_20260914.csv) · [全部逐基因 PCC](../benchmarks/liver/optimization_per_gene.csv) · [此前七基线及消融](../benchmarks/liver/baseline_20260913.csv)。
- [训练前方案](../benchmarks/liver/optimization_plan.json) · [C1 选择记录](../benchmarks/liver/optimization_selection_locked.json)。选择记录先于本批 D1 RNA 评价写入，但不代表 D1 在项目历史中从未被查看。
- 本地 `results/optimization_20260914/` 保留权重、完整历史、各候选预测和检查记录；`prediction_only.npz` 与 `predicted_log_expression.csv.gz` 是不含实测 RNA 的最终表达预测。
- 仓库入口已统一为 `run.py`，核心代码放在 `src/he2st/`，参数放在 `configs/`。数据/权重/论文/PPT不进入 Git，历史编号脚本保留索引和原复现路径。

详细前轮诊断与论文口径核对见 [06 报告](06_LIVER_CONTEXT_IMPROVEMENT.md) 和 [05 报告](05_PCC_DIAGNOSIS.md)。运行方式见 [快速开始](QUICKSTART.md)。
'''
    (ROOT/'docs/RESULTS.md').write_text(report, encoding='utf8')
    readme = ROOT/'README.md'
    text = readme.read_text(encoding='utf8')
    start = text.index('**固定协议：')
    end = text.index('\n\n', start)
    text = text[:start]+'**固定协议：A1+B1 训练、C1 验证、D1 测试；训练集选定 HEG200 / HEG50；测试真值不做重建。** '+f'本轮 C1 选择模型在固定 D1 的 PCC 为 **{best["HEG200"]:.4f} / {best["HEG50"]:.4f}**，原 ST-Net 为 **0.1472 / 0.1978**。完整对照和局限见[结果报告](docs/RESULTS.md)；固定面板平均仍未达到 0.5。'+text[end:]
    readme.write_text(text, encoding='utf8')
    paths = sorted((ROOT/'src/he2st').glob('*.py'))+sorted((ROOT/'configs').glob('*.json'))
    snapshot = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (public/'source_snapshot.json').write_text(json.dumps(snapshot, indent=2), encoding='utf8')
    print('Published result snapshots:', select['selected'], best)


if __name__ == '__main__':
    main()
