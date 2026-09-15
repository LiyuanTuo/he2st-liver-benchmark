"""Create explanatory report figures from source papers and frozen experiments.

Requires local papers/data/results only for regeneration. Curated figure files
are committed so compiling the report does not require those large inputs.
"""
from pathlib import Path
import gzip
import hashlib
import json
import shutil
import fitz
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT/'report/figures'
PUBLIC = ROOT/'benchmarks/liver'
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42})


def save(fig, name):
    fig.savefig(FIG/(name+'.pdf'), bbox_inches='tight')
    plt.close(fig)


def architecture_sources():
    sources = []
    for name, paper, page, rect, citation, url in [
        ('genar_arch', 'GenAR_2026_Medical_Image_Analysis_arXiv.pdf', 3, (108,79,503,347), 'genar', 'https://arxiv.org/abs/2510.04315'),
        ('ressat_arch', 'ResSAT.pdf', 24, (100,61,513,273), 'ressat', 'https://doi.org/10.1186/s13059-026-04168-x')]:
        src = ROOT/'papers'/paper
        with fitz.open(src) as doc:
            doc[page].get_pixmap(matrix=fitz.Matrix(3,3), clip=fitz.Rect(rect)).save(FIG/(name+'.png'))
        sources.append({'figure': name+'.png', 'source': 'papers/'+paper, 'pdf_page_1based': page+1,
                        'crop_points': rect, 'source_sha256': hashlib.sha256(src.read_bytes()).hexdigest(),
                        'citation_key': citation, 'url': url})
    for name, cite, url in [
        ('stnet_arch','stnet','https://doi.org/10.1038/s41551-020-0578-x'),
        ('bleep_arch','bleep','https://github.com/bowang-lab/BLEEP'),
        ('stem_arch','stem','https://github.com/SichenZhu/Stem')]:
        src = ROOT/'figures/ppt_assets'/(name+'.png')
        shutil.copyfile(src, FIG/src.name)
        sources.append({'figure': src.name, 'source': str(src.relative_to(ROOT)).replace('\\','/'),
                        'source_sha256': hashlib.sha256(src.read_bytes()).hexdigest(), 'citation_key': cite,
                        'url': url, 'extraction': 'Existing stage presentation asset; ST-Net Fig. 1a, BLEEP overview, Stem Fig. 1'})
    (FIG/'sources.json').write_text(json.dumps(sources, indent=2)+'\n', encoding='utf8')


def diagram(name, blocks, arrows, ylim, footer=None):
    fig, ax = plt.subplots(figsize=(10, ylim*.62), layout='constrained')
    ax.set(xlim=(0,10), ylim=(0,ylim)); ax.axis('off')
    for x,y,w,h,txt,color in blocks:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.06',fc=color,ec='#8090a0'))
        ax.text(x+w/2,y+h/2,txt,ha='center',va='center',fontsize=10)
    for a,b in arrows:
        ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',lw=1.4,color='#475569'))
    if footer: ax.text(5,.15,footer,ha='center',va='bottom',fontsize=9)
    save(fig,name)


def main():
    FIG.mkdir(exist_ok=True)
    architecture_sources()
    fig, axes = plt.subplots(2,2,figsize=(9,6),layout='constrained')
    for ax,s,n,role in zip(axes.flat,'ABCD',[2378,2349,2277,2265],['Train','Train','Validation','Test']):
        path = next((ROOT/'data/raw/GSE240429').glob(f'*_C73{s}1_tissue_lowres_image.png.gz'))
        with gzip.open(path,'rb') as f: im = np.asarray(Image.open(f).convert('RGB'))
        ax.imshow(im); ax.axis('off'); ax.set_title(f'C73_{s}1 | {role} | {n:,} spots')
    save(fig,'data_overview')

    genes = pd.read_csv(PUBLIC/'genes.csv').sort_values('train_expression_rank')
    fig, axes = plt.subplots(1,2,figsize=(9,3.2),layout='constrained')
    for ax,field,label in zip(axes,['train_mean_log1p_all_genes','train_detection_fraction'],['Mean whole-transcriptome log1p','Fraction of training spots detected']):
        ax.plot(genes.train_expression_rank,genes[field],lw=1,color='#365b83')
        ax.axvspan(1,50,alpha=.17,color='#c7654b',label='Fixed HEG50')
        ax.set(xlabel='Training expression rank (HEG200)',ylabel=label); ax.legend(fontsize=8)
    save(fig,'gene_panel')

    diagram('protocol_flow',[
        (.2,3.4,2.6,1.1,'A1 + B1 RNA\nRank mean expression\nFreeze HEG200 / HEG50','#edf5ef'),
        (3.6,3.4,2.7,1.1,'A1 + B1 H&E / RNA\nFit model parameters\nTrain-only target scaling','#edf5ef'),
        (7.1,3.4,2.6,1.1,'C1 H&E / RNA\nSelect epoch, TTA\nand ensemble weight','#fff0d7'),
        (.2,1.0,2.6,1.1,'D1 H&E + XY\nThree image views\nFrozen gene metadata','#eaf2f8'),
        (3.6,1.0,2.7,1.1,'Six frozen models\n0.75 O + 0.25 N\n2265 x 200 prediction','#eaf2f8'),
        (7.1,1.0,2.6,1.1,'D1 measured RNA\nPanel log1p target\nPer-gene PCC + MAE','#f8e8e7')],
        [((2.8,3.95),(3.6,3.95)),((6.3,3.95),(7.1,3.95)),((2.8,1.55),(3.6,1.55)),((6.3,1.55),(7.1,1.55)),((8.4,3.4),(5.0,2.1))],5.1,
        'Top: fitting / selection. Bottom: frozen inference / evaluation. RNA enters only the scoring box on D1.')

    diagram('contextfusion_arch',[
        (.15,4.25,2.0,.85,'Local view\n224 px / 58 um','#edf3f8'),
        (.15,2.85,2.0,.85,'Near context\n896 px / 231 um','#edf3f8'),
        (.15,1.45,2.0,.85,'Wide context\n1792 px / 461 um','#edf3f8'),
        (2.8,1.5,2.15,3.55,'Resize each to\n224 x 224 RGB\n\nShared CNN encoder\nResNet18: d = 512\nEfficientNet-B0:\nd = 1280','#edf5ef'),
        (5.6,1.5,1.55,3.55,'Concatenate\n3 view features\n\n[B, 3d]\nLayerNorm\nDropout 0.3','#fff0d7'),
        (7.8,1.5,2.0,3.55,'Linear: 3d -> 256\nGELU\nDropout 0.2\nLinear: 256 -> 200\n\nInvert train scaling\nPanel-log prediction','#f8e8e7')],
        [((2.15,4.68),(2.8,4.68)),((2.15,3.28),(2.8,3.28)),((2.15,1.88),(2.8,1.88)),((4.95,3.3),(5.6,3.3)),((7.15,3.3),(7.8,3.3))],5.6,
        'Training: standardized MSE + 0.2 (1 - batch gene PCC). One encoder shares weights across all three views.')

    meta=np.load(ROOT/'data/processed/gse240429_heg/arrays/C73_A1.npz')
    xy=meta['coordinates_xy']; spot=int(np.argmin(((xy-np.median(xy,axis=0))**2).sum(1)))
    fig,axes=plt.subplots(1,3,figsize=(9,3.1),layout='constrained')
    for ax,scale in zip(axes,[224,896,1792]):
        patch=np.load(ROOT/f'data/processed/gse240429/context_20260913/C73_A1_{scale}_patches.npy',mmap_mode='r')[spot]
        ax.imshow(patch);ax.set_title(f'{scale} px field -> 224 px input');ax.axis('off')
    save(fig,'multiscale_patches')

    names=['resnet18_ema','efficientnet_b0_ema','resnet18_spatial']
    labels=['ResNet18','EfficientNet-B0','R18 + label smoothing']
    fig,axes=plt.subplots(1,3,figsize=(10,3),layout='constrained')
    for ax,name,label in zip(axes,names,labels):
        history=json.loads((ROOT/f'results/optimization_20260914/{name}/history.json').read_text())
        for mode,style in [('raw','-'),('ema','--')]:
            ax.plot([r['epoch'] for r in history],[r['validation'][mode]['HEG200'] for r in history],style,label=mode)
        ax.set(title=label,xlabel='Epoch',ylabel='C1 HEG200 PCC',ylim=(.12,.37));ax.legend(fontsize=8)
    save(fig,'validation_curves')

    d=dict(np.load(ROOT/'results/optimization_20260914/test_candidates.npz'))
    def corr(p):
        a=p.astype(float)-p.mean(0,dtype=float);b=d['truth'].astype(float)-d['truth'].mean(0,dtype=float)
        return (a*b).sum(0)/np.sqrt((a*a).sum(0)*(b*b).sum(0))
    prior=corr(d['previous_contextfusion']);final=corr(d['new_weight_0.25'])
    ordered=pd.read_csv(PUBLIC/'genes.csv');np.testing.assert_array_equal(ordered.gene,d['genes'])
    mask=ordered.is_primary_HEG50.values
    fig,axes=plt.subplots(1,3,figsize=(11,3.1),layout='constrained')
    ax=axes[0];bins=np.linspace(-.3,.9,25)
    ax.hist(final[~mask],bins=bins,alpha=.7,label='Other HEG150');ax.hist(final[mask],bins=bins,alpha=.7,label='HEG50');ax.axvline(.5,ls='--',color='gray');ax.set(xlabel='Final per-gene D1 PCC',ylabel='Number of genes');ax.legend(fontsize=8)
    axes[1].scatter(prior,final,s=12,c=np.where(mask,'#c7654b','#365b83'),alpha=.7);axes[1].plot([-.3,.9],[-.3,.9],c='gray',ls='--');axes[1].set(xlabel='Prior ensemble D1 PCC',ylabel='Final ensemble D1 PCC',xlim=(-.3,.9),ylim=(-.3,.9))
    axes[2].scatter(ordered.train_mean_log1p_all_genes,final,s=12,c=np.where(mask,'#c7654b','#365b83'),alpha=.7);axes[2].set(xlabel='Train mean log1p (all-gene denominator)',ylabel='Final D1 PCC')
    save(fig,'per_gene_diagnostics')
    stats={'figure_context': 'post-hoc diagnostics; all fixed genes retained',
           'genes_above_0.5_HEG200':int((final>.5).sum()),'genes_above_0.5_HEG50':int((final[mask]>.5).sum()),
           'improved_over_prior':int((final>prior).sum()),'worse_than_prior':int((final<prior).sum()),
           'train_expression_vs_D1_PCC_spearman':float(spearmanr(ordered.train_mean_log1p_all_genes,final).statistic),
           'HEG200_detection_min':float(genes.train_detection_fraction.min()),
           'HEG200_detection_median':float(genes.train_detection_fraction.median()),
           'multiscale_example_A1_barcode':str(meta['barcodes'][spot])}
    (PUBLIC/'report_figure_diagnostics.json').write_text(json.dumps(stats,indent=2)+'\n')

    matrix=pd.read_csv(PUBLIC/'ensemble_residual_correlation.csv',index_col=0)
    fig,ax=plt.subplots(figsize=(6.2,4.7),layout='constrained');im=ax.imshow(matrix,vmin=.97,vmax=1,cmap='YlOrRd')
    ticks=['Old s42','Old s17','Old s83','New R18','New B0','R18 smooth']
    ax.set_xticks(range(6),ticks,rotation=35,ha='right');ax.set_yticks(range(6),ticks)
    for i in range(6):
        for j in range(6):ax.text(j,i,f'{matrix.iloc[i,j]:.3f}',ha='center',va='center',fontsize=9,color='white' if matrix.iloc[i,j]>.99 else 'black')
    fig.colorbar(im,ax=ax,label='Flattened spot-gene residual correlation');save(fig,'residual_correlation')

    val=pd.read_csv(PUBLIC/'report_spatial_ablation/validation.csv')
    fig,ax=plt.subplots(figsize=(7,3),layout='constrained')
    for k in [6,12,24]:
        rows=pd.concat([val.iloc[:1],val[val.k==k]])
        ax.plot(rows.alpha,rows.HEG200,'o-',label=f'k = {k}')
    ax.set(xlabel='Neighbor prediction weight alpha',ylabel='C1 HEG200 PCC');ax.legend();save(fig,'prediction_smoothing')
    print(json.dumps(stats,indent=2))


if __name__=='__main__':
    main()
