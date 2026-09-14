"""Recompute, compare, and report new experiments against identical old targets."""
from pathlib import Path
import hashlib,json,importlib.util
import numpy as np
import pandas as pd
from scipy.stats import pearsonr,spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/liver_context_20260913';FIG=ROOT/'figures/liver_context_20260913';FIG.mkdir(parents=True,exist_ok=True)
spec=importlib.util.spec_from_file_location('diag',ROOT/'scripts/60_diagnose_liver_signal.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False})
def read(p):return json.loads((ROOT/p).read_text(encoding='utf8'))
def metric(y,p,mask):
    r=mod.pcc(y.astype(float),p.astype(float))
    return dict(HEG200=float(np.mean(r)),HEG50=float(np.mean(r[mask])),MAE=float(np.abs(y-p).mean()),genes_above_05=int((r>.5).sum()),HEG50_above_05=int((r[mask]>.5).sum()))
def main():
    old=dict(np.load(ROOT/'results/verified_20260913/benchmark_predictions.npz'))
    y=old['truth'];genes=old['genes'];xy=old['coordinates_xy'];bars=old['barcodes']
    audit=pd.read_csv(ROOT/'results/verified_20260913/fixed_HEG200_gene_audit.csv');mask=audit.is_primary_HEG50.to_numpy(bool)
    np.testing.assert_array_equal(audit.gene,genes)
    predictions={n:old[n] for n in read('results/verified_20260913/benchmark.json')['methods']}
    frozen=dict(np.load(OUT/'D1_predictions.npz'))
    frozen_sel=read('results/liver_context_20260913/selection_locked.json')['selected']
    predictions['Context-RBF']=frozen[frozen_sel]
    newbase=ROOT/'results/liver_context_fusion_20260913'
    for arm in ['imagenet','scratch']:
        d=dict(np.load(newbase/f'{arm}_D1.npz'))
        np.testing.assert_array_equal(d['genes'],genes);np.testing.assert_array_equal(d['barcodes'],bars)
        np.testing.assert_allclose(d['truth'],y,atol=1e-6)
        predictions['ContextFusion-'+arm]=d['predicted']
    d=dict(np.load(ROOT/'results/liver_single_view_control_20260913/imagenet_D1.npz'))
    np.testing.assert_array_equal(d['genes'],genes);np.testing.assert_array_equal(d['barcodes'],bars)
    np.testing.assert_allclose(d['truth'],y,atol=1e-6)
    predictions['SingleView-control']=d['predicted']
    d=dict(np.load(ROOT/'results/liver_context_ensemble_20260913/selected_D1.npz'))
    for key,value in [('genes',genes),('barcodes',bars),('coordinates_xy',xy)]:np.testing.assert_array_equal(d[key],value)
    np.testing.assert_allclose(d['truth'],y,atol=1e-6)
    predictions['ContextFusion-multiseed']=d['predicted']
    rows=[]
    for name,p in predictions.items():
        assert p.shape==y.shape==(2265,200) and np.isfinite(p).all()
        r=mod.pcc(y.astype(float),p.astype(float));assert np.isfinite(r).all()
        for j in [0,25,100,199]:np.testing.assert_allclose(r[j],pearsonr(y[:,j].astype(float),p[:,j].astype(float)).statistic,atol=1e-12)
        rows.append(dict(method=name,**metric(y,p,mask)))
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'combined_benchmark.csv',index=False)
    pd.DataFrame(dict(gene=genes,HEG50=mask,**{n:mod.pcc(y,p) for n,p in predictions.items()})).to_csv(OUT/'combined_per_gene.csv',index=False)
    np.savez_compressed(OUT/'combined_predictions.npz',truth=y,genes=genes,barcodes=bars,coordinates_xy=xy,**predictions)
    vals={f'ContextFusion-{arm}':read(f'results/liver_context_fusion_20260913/{arm}_selection.json')['validation']['HEG200'] for arm in ['imagenet','scratch']}
    vals['SingleView-control']=read('results/liver_single_view_control_20260913/imagenet_selection.json')['validation']['HEG200']
    vals['Context-RBF']=read('results/liver_context_20260913/selection_locked.json')['validation'][frozen_sel]['validation']['HEG200']
    ensemble_selection=read('results/liver_context_ensemble_20260913/selection_locked.json')
    seed_results=read('results/liver_context_ensemble_20260913/test_metrics.json')['models']
    seed_values=np.array([[seed_results[f'seed{s}'][k] for k in ['HEG200','HEG50']] for s in [42,17,83]])
    vals['ContextFusion-multiseed']=ensemble_selection['validation'][ensemble_selection['selected']]['HEG200']
    selected=max(vals,key=vals.get)
    c1path={'ContextFusion-imagenet':'liver_context_fusion_20260913/imagenet_C1.npz',
      'ContextFusion-scratch':'liver_context_fusion_20260913/scratch_C1.npz',
      'SingleView-control':'liver_single_view_control_20260913/imagenet_C1.npz',
      'ContextFusion-multiseed':'liver_context_ensemble_20260913/selected_C1.npz'}
    if selected in c1path:
        c1=np.load(ROOT/'results'/c1path[selected]);validation_r=mod.pcc(c1['truth'],c1['predicted'])
    else:
        c1=np.load(OUT/'C1_predictions.npz');validation_r=mod.pcc(c1['truth'],c1[frozen_sel])
    signal=pd.read_csv(OUT/'train_validation_signal_diagnostics.csv');signal=signal[signal.slide=='C1'].copy()
    np.testing.assert_array_equal(signal.gene,genes)
    signal['validation_model_pcc']=validation_r
    rho_moran=float(spearmanr(signal.moran6,validation_r).statistic)
    rho_count=float(spearmanr(signal.mean_count,validation_r).statistic)
    signal.to_csv(OUT/'validation_predictability_diagnosis.csv',index=False)
    pd.DataFrame(predictions[selected],index=bars,columns=genes).rename_axis('barcode').to_csv(OUT/'D1_predicted_panel_log1p.csv.gz',compression='gzip')
    (OUT/'prediction_units.json').write_text(json.dumps(dict(model=selected,units='Predicted log1p(1e4 * HEG200-relative abundance)',
       not_raw_UMI_counts=True,spot_order='barcode rows',gene_order='fixed training HEG200 columns',
       expression_truth_modified=False),indent=2))
    # Paired spatial block resampling, uncertainty conditional on this ONE slide.
    blocks=np.minimum(((xy-xy.min(0))/(np.ptp(xy,axis=0)+1e-9)*5).astype(int),4)
    blockids=blocks[:,0]+5*blocks[:,1];groups=[np.flatnonzero(blockids==i) for i in np.unique(blockids)]
    rng=np.random.default_rng(42);boot=[]
    for b in range(300):
        ix=np.concatenate([groups[i] for i in rng.integers(0,len(groups),len(groups))])
        a=metric(y[ix],predictions[selected][ix],mask);base=metric(y[ix],predictions['ST-Net (DenseNet121)'][ix],mask)
        boot.append([a['HEG200'],a['HEG50'],a['HEG200']-base['HEG200'],a['HEG50']-base['HEG50']])
    ci=np.quantile(boot,[.025,.975],axis=0)
    stats=dict(selected_by_C1=selected,C1_candidates=vals,spatial_blocks=len(groups),bootstrap_repeats=300,
      CI95=dict(zip(['HEG200','HEG50','delta_HEG200_vs_STNet','delta_HEG50_vs_STNet'],ci.T.tolist())),
      seed_mean=seed_values.mean(0).tolist(),seed_std=seed_values.std(0,ddof=1).tolist(),
      validation_predictability_spearman=dict(moran6=rho_moran,mean_count=rho_count),
      interpretation='Paired spatial block bootstrap conditional on one donor/slide; not independent-patient uncertainty.',
      metrics=frame.set_index('method').to_dict('index'))
    (OUT/'comparison.json').write_text(json.dumps(stats,indent=2))
    names=list(predictions);colors=['#a9b7c4']*7+['#c2a675','#b84842','#608c6e','#8d78aa','#9c2829']
    fig,ax=plt.subplots(1,2,figsize=(13,6),sharey=True)
    for axis,key,title in zip(ax,['HEG200','HEG50'],['固定 200 个高表达基因','固定前 50 个高表达基因']):
        values=frame[key].to_numpy();axis.barh(names,values,color=colors)
        for i,v in enumerate(values):axis.text(v+.008,i,f'{v:.3f}',va='center',fontsize=9)
        axis.axvline(.5,color='#922b29',ls='--',lw=1);axis.set_xlim(0,.58);axis.set_title(title);axis.set_xlabel('D1 每基因 PCC 的平均值')
    ax[0].invert_yaxis();fig.tight_layout();fig.savefig(FIG/'benchmark.png',dpi=170);plt.close(fig)
    # Prespecified examples remain expression rank 1 and 2, not best test genes.
    examples=read('results/verified_20260913/panel_audit.json')['primary_top50'][:2]
    map_methods=['Truth','ST-Net (DenseNet121)',selected,'ContextFusion-scratch']
    fig,axes=plt.subplots(2,4,figsize=(14,6))
    for row,gene in enumerate(examples):
        j=list(genes).index(gene);lo,hi=np.quantile(y[:,j],[.01,.99])
        for col,name in enumerate(map_methods):
            p=y if name=='Truth' else predictions[name];axis=axes[row,col]
            dots=axis.scatter(xy[:,0],xy[:,1],c=p[:,j],s=4,cmap='viridis',vmin=lo,vmax=hi,linewidths=0)
            title=f'{gene} / {name}'
            if name!='Truth':title+=f'\nr={mod.pcc(y,p)[j]:.3f}'
            axis.set_title(title,fontsize=10);axis.set_aspect('equal');axis.invert_yaxis();axis.axis('off')
        colorax=fig.add_axes([.93,.58 if row==0 else .12,.012,.28])
        colorbar=fig.colorbar(dots,cax=colorax);colorbar.ax.tick_params(labelsize=8);colorbar.set_label('面板 log1p',fontsize=8)
    fig.subplots_adjust(left=.02,right=.91,bottom=.035,top=.91,wspace=.08,hspace=.30)
    fig.savefig(FIG/'ALB_HP_fixed_examples.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(12,4))
    for arm in ['imagenet','scratch']:
        history=read(f'results/liver_context_fusion_20260913/{arm}_training.json')
        for axis,key in zip(ax,['HEG200','HEG50']):axis.plot([r['epoch'] for r in history],[r['validation'][key] for r in history],label=arm)
    for axis,key in zip(ax,['HEG200','HEG50']):axis.set(xlabel='训练轮数',ylabel=f'C1 {key} PCC');axis.axhline(.5,c='gray',ls='--');axis.legend();axis.grid(alpha=.2)
    fig.tight_layout();fig.savefig(FIG/'learning_curves.png',dpi=160);plt.close(fig)
    best=stats['metrics'][selected];base=stats['metrics']['ST-Net (DenseNet121)'];scratch=stats['metrics']['ContextFusion-scratch']
    diag=read('results/liver_context_20260913/signal_diagnostics.json');reg=read('results/liver_registration_20260913/test_metrics.json')
    thinning=read('results/liver_context_20260913/count_thinning_ablation.json')
    checkpoint_audit=read('results/liver_context_20260913/fresh_checkpoint_verification.json')
    assert len(checkpoint_audit)==5,'All five checkpoints must finish before the final report'
    train_rows='| 模型 | A1+B1训练 HEG200/HEG50 | C1验证 HEG200/HEG50 | D1测试 HEG200/HEG50 |\n|---|---:|---:|---:|\n'
    for arm in ['imagenet','scratch']:
        item=checkpoint_audit['liver_context_fusion_20260913/'+arm]
        train_rows+='| '+arm+' | '+' | '.join(f'{item[k]["HEG200"]:.4f}/{item[k]["HEG50"]:.4f}' for k in ['train_metrics','validation_metrics','archived_metrics'])+' |\n'
    max_recheck_pcc=max(max(v['absolute_PCC_difference'].values()) for v in checkpoint_audit.values())
    table='| 方法 | HEG200 PCC | HEG50 PCC | MAE | PCC>0.5 基因数/200 |\n|---|---:|---:|---:|---:|\n'
    for row in rows:table+=f'| {row["method"]} | {row["HEG200"]:.4f} | {row["HEG50"]:.4f} | {row["MAE"]:.4f} | {row["genes_above_05"]} |\n'
    text=f'''# 人肝 PCC 深入诊断与新模型实测（2026-09-13）

已完成三尺度端到端模型、全随机初始化训练、同配方单视野消融，以及冻结特征/邻域/核回归控制。

**按 C1 验证集选中的新模型为 {selected}。D1 固定 HEG200 PCC={best['HEG200']:.4f}，HEG50 PCC={best['HEG50']:.4f}；此前 ST-Net 为 {base['HEG200']:.4f}/{base['HEG50']:.4f}。**
相对原 ST-Net 的绝对变化分别为 {best['HEG200']-base['HEG200']:+.4f} / {best['HEG50']-base['HEG50']:+.4f}。
两个平均值是否达到0.5：HEG200={best['HEG200']>=.5}；HEG50={best['HEG50']>=.5}。不能以个别高分基因代替面板平均。

## 1. 不变的评测条件

- 同一供者 C73；A1+B1 训练（4727 spots）、C1 验证（2277）、D1 测试（2265）。
- 同一训练集选定的 HEG200，HEG50 是相同表达排序中的前50，未按测试成绩换基因。
- 真值保持原始实测数据的 `log1p(10000*count/HEG200总count)`，不做 PCA/Harmony/邻域平滑；每基因跨 spot 计算 Pearson，再取固定面板平均。
- 新回归网络直接输出该 log 表达单位；原方法的原始计数/全转录组 log 输出先换算到该单位。新网络的预测均值不要求满足 count 组成闭合约束，不对它再次归一化。真值和 PCC 算法完全一致。
- 新模型的参数、epoch、TTA 由 C1 HEG200 PCC 选择，`selection_locked.json` 先落盘，再打开 D1 RNA。D1 在之前研究中已被评测，故属于反复开发的测试切片，不能称全新外部验证。
- 统一的是切片划分、基因、目标和指标。方法的图像上下文、编码器、预训练与计算预算仍不同；本表不能证明严格等预算下某架构普遍更优，额外视野的作用另用同配方单视野控制衡量。

## 2. 同条件结果

{table}

MAE 也在同一 log 表达单位。新方法中最终推荐以 C1 的选择为准，未选择 D1 最高的一行。

![结果](../figures/liver_context_20260913/benchmark.png)

## 3. 原因与改进：哪些已有证据

### 3.1 不能把所有低分都归因于选错基因

旧面板与 HEG200 无重叠、D1 零值比例24.70%，这是上一轮已纠正的问题。当前固定高表达面板零值3.45%，其余差距不能继续用同一个错误解释。

### 3.2 训练目标与最终计分目标不一致，确实有损失，但不是唯一原因

原有多数模型训练全转录组分母的 log 表达，再转换到200基因分母评测。对每个 spot，这引入由面板占总转录本比例决定的变化。新实验直接训练最终目标。
相同 ResNet18 单视野特征及搜索预算，Ridge 的 D1 HEG200 从0.0937变为0.0982；三视野从0.1018变为0.1192。
因此该修正确有改善，但远不足以单独解释或解决到0.5的差距。详见 `validation.json`、`test_metrics.json` 中全部配置；各目标独立按相同 C1 搜索选择正则与平滑。

### 3.3 单 spot 视野遗漏组织结构；同时需要端到端调整图像特征

原224像素约58微米，与55微米 spot 相当。新模型增加896/1792像素视野，约231/461微米，学习更广的组织结构。
冻结 ImageNet 特征仅加多尺度/图邻域/复杂回归头，没有超过原 ST-Net；这否定了“只拼接更多特征就够了”的简单假设。
新 ContextFusion 三个尺度共享 ResNet18，再用 LayerNorm、256维 GELU 回归头联合输出200基因；训练集逐基因标准化目标，优化 MSE+0.2×相关性损失，全部编码器可训练。
`SingleView-control` 将同一个224像素图块重复三次，维持参数量、损失、优化器和计算流程，单独检验额外视野。它与三视野的具体差值见表，不能把同时更换训练配方的全部提升都归于视野。
这一严格视野对照的 D1 HEG200/HEG50 为 {stats['metrics']['SingleView-control']['HEG200']:.4f}/{stats['metrics']['SingleView-control']['HEG50']:.4f}；对应三视野预训练seed42为 {stats['metrics']['ContextFusion-imagenet']['HEG200']:.4f}/{stats['metrics']['ContextFusion-imagenet']['HEG50']:.4f}。
额外视野带来的绝对增量为 {stats['metrics']['ContextFusion-imagenet']['HEG200']-stats['metrics']['SingleView-control']['HEG200']:+.4f}/{stats['metrics']['ContextFusion-imagenet']['HEG50']-stats['metrics']['SingleView-control']['HEG50']:+.4f}。单视野新训练配方本身已经超过旧ST-Net；其中仍同时包含编码器、回归头和损失的差异，不能将它归于某一个唯一因素。

### 3.4 从零训练已真实执行

`ContextFusion-scratch` 的编码器和预测头全部随机初始化，未加载 ImageNet 或以前的本项目 checkpoint。
D1 HEG200={scratch['HEG200']:.4f}，HEG50={scratch['HEG50']:.4f}。`ContextFusion-imagenet` 则明确使用 ImageNet 初始化，并在 A1+B1 重训所有参数；两者不能混称为全随机从零训练。
初始比较固定 seed42；三尺度预训练模型另重复 seed17、83。每次最多60轮、C1 连续12轮无改善早停，旋转/翻转/轻微亮度对比度增强。完整训练曲线、权重、选择记录和预测矩阵均保存。
预训练编码器学习率3e-5、随机编码器3e-4，预测头均为3e-4；因此预训练/全随机比较包含合适训练学习率的差异，不是只替换初始化的严格因果消融。
多种子版本在三个单次预测与等权平均之间，仅由 C1 选择；本次选择 `{ensemble_selection['selected']}`。没有按D1学习集成权重。三次 D1 成绩完整保存在 `results/liver_context_ensemble_20260913/test_metrics.json`。
三个预训练种子的D1均值±样本标准差：HEG200 {seed_values[:,0].mean():.4f}±{seed_values[:,0].std(ddof=1):.4f}；HEG50 {seed_values[:,1].mean():.4f}±{seed_values[:,1].std(ddof=1):.4f}。这些重复仍来自同一供者、同一划分，不是三个独立患者。

![验证曲线](../figures/liver_context_20260913/learning_curves.png)

### 3.5 高表达不等于跨空间可预测；0.5不是已知的数据集下限

C1 的6邻居 Moran's I：HEG200平均0.1373、HEG50平均0.2414。直接使用 C1 邻居的实测 RNA 做**标签侧诊断**，PCC为0.2602/0.4176。
在全部固定200基因中，选中模型的 C1 逐基因 PCC 与 Moran's I 的 Spearman相关为 {rho_moran:.3f}，与平均计数的相关为 {rho_count:.3f}。这是基因间观察关联，不是因果证明，也不用于重新选择基因。
把同一计数随机拆成两个半深度样本，20次拆分的相关性平均0.2719/0.4655。该实验条件于观测计数，既不是真实技术重复，也不能当作模型性能的硬上限。
这些结果提示空间变化、计数噪声与形态可辨认性都应考虑；并不能证明0.5不可能。不能为了达到目标而平滑测试真值、只汇报最好预测的基因或改变平均轴。

2026年同一 GSE240429 的研究报告 EfficientNet-B0 的 HEG 平均 PCC 0.310；它使用 BLEEP 的基因并集/Harmony流程、slide3测试，不能直接当作我们D1固定HEG面板的可比成绩。[原论文](https://link.springer.com/article/10.1186/s12859-026-06447-7)
ResSAT 0.6577/0.6980 是另一数据和重建目标的2000HVG指标，原始复现已另行完成，详见[上轮核验](05_PCC_DIAGNOSIS.md)。

### 3.6 C1 与 D1 捕获计数相差约一倍，能解释部分验证—测试落差

已经与 h5ad 的原始 `total_counts_all_genes` 独立核对：每spot总UMI中位数 A1=8134、B1=6537、C1=12423、D1=6010。D1仅为C1的48.4%。
固定200基因的逐基因平均spot计数，再在基因间取中位数，C1约15.45，D1约9.18；“高表达”的相对排名不等于无计数噪声。
保持现有预测不变，对C1计数副本以0.48378概率保留UMI，重复30次；主D1真值始终未改。

| 固定模型 | 原C1 HEG200/HEG50 | 降采样C1平均 HEG200/HEG50 |
|---|---:|---:|
| 三尺度预训练seed42 | 0.3485 / 0.4854 | {thinning['models']['imagenet']['thinned_C1_mean'][0]:.4f} / {thinning['models']['imagenet']['thinned_C1_mean'][1]:.4f} |
| 三尺度全随机seed42 | 0.3038 / 0.4385 | {thinning['models']['scratch']['thinned_C1_mean'][0]:.4f} / {thinning['models']['scratch']['thinned_C1_mean'][1]:.4f} |

因此，减少可用计数本身就降低PCC，但仍不能解释全部D1下降。捕获UMI总数同时受到细胞RNA含量、捕获效率和测序等影响；降采样是代理实验，不能把它全部归因为测序深度。该诊断未用于进一步选择模型或更改测试真值。完整30次统计见 `count_thinning_ablation.json`。

### 3.7 已能拟合训练信号，主要仍需改善跨切片泛化

从保存的checkpoint独立重新推理全部训练spots（不做随机增强）得到：

{train_rows}

例如，预训练模型训练HEG50已经达到约0.608，D1却只有0.302。这显示当前模型有拟合能力，但不能把训练集较高相关性当作未见切片预测能力。仅单一供者的两个训练切片，也不能把4727个spot当成4727个独立生物学重复。
PCC提升还不等于绝对表达水平准确：选中集成对D1的ALB预测均值约7.057，真值7.420；HP预测均值6.141，真值5.844，均在相同面板log单位。PCC对每基因的常数偏差不敏感，故它们不能解释所有低相关性，但提醒我们同时检查MAE和表达校准，不仅追求PCC。

## 4. 辅助反证与边界

只用 H&E 配准连续切片，再从 A1/B1 参考 RNA 迁移，C1 选择 A1+B1 的12邻居控制；D1 PCC={reg['metrics']['HEG200']:.4f}/{reg['metrics']['HEG50']:.4f}。粗尺度图像 ECC 为0.98–0.99，但这不等于 spot 级 RNA 一致；局部形变、细胞差异与技术差异仍未分离。
这是依赖同供者连续参考切片的控制，单独报告，不将其包装成从头深度学习预测。配准完全不使用 C1/D1 RNA。

选中模型的 D1 空间块 bootstrap 95%区间：HEG200 {ci[0,0]:.4f}–{ci[1,0]:.4f}，HEG50 {ci[0,1]:.4f}–{ci[1,1]:.4f}。
与 ST-Net 的配对差值区间：HEG200 {ci[0,2]:+.4f}–{ci[1,2]:+.4f}，HEG50 {ci[0,3]:+.4f}–{ci[1,3]:+.4f}。
该区间仅反映一张切片内的抽样变动，不能代替多患者外部验证，也没有消除开发期多次实验的影响。

![预先固定实例](../figures/liver_context_20260913/ALB_HP_fixed_examples.png)

## 5. 复核与运行

- 新结果：`results/liver_context_20260913/combined_benchmark.csv`、`combined_per_gene.csv`、`combined_predictions.npz`。
- 端到端：`results/liver_context_fusion_20260913/`；单视野控制：`results/liver_single_view_control_20260913/`。
- 图像缓存：`data/processed/gse240429/context_20260913/`，仅 H&E 和坐标参与提取。
- 在 WSL 全局 Python 顺序运行 `scripts/57_extract_liver_context.py`、`58_liver_context_experiments.py`、`59_train_context_fusion.py`、`63_train_single_view_control.py`、`64_context_seed_repeats.py`、`65_context_ensemble.py`。
- Windows 全局 Python 3.12 运行 `60_diagnose_liver_signal.py`、`62_report_liver_context.py`；61配准控制在WSL执行。
- 70计数诊断在WSL执行；69可导出仅含预测表达、坐标和H&E叠加元数据的AnnData文件，不混入D1实测RNA。
- 预测矩阵逐一核对2265×200、基因/条形码对齐、有限值、相同真值；PCC抽样与SciPy独立重算一致。
- 五个CNN checkpoint均重新推理全部D1 spots，原存档与重算的HEG200/HEG50平均PCC最大绝对差为 {max_recheck_pcc:.8f}。BF16/cuDNN初次核验出现小数值差异，后续匹配输入布局与单视野运算顺序、关闭cuDNN自动选算法，并比较全切片指标；不笼统声称逐位一致。完整数值差异和FP32参照在 `fresh_checkpoint_verification.json`。
'''
    (ROOT/'docs/06_LIVER_CONTEXT_IMPROVEMENT.md').write_text(text,encoding='utf8')
    paths=[OUT/'combined_predictions.npz',OUT/'comparison.json',ROOT/'docs/06_LIVER_CONTEXT_IMPROVEMENT.md']
    verification=dict(shape=[2265,200],fixed_genes=True,truth_unchanged=True,scipy_pcc_checked=True,
       sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    (OUT/'verification.json').write_text(json.dumps(verification,indent=2))
    print(json.dumps(stats,indent=2))

if __name__=='__main__':main()
