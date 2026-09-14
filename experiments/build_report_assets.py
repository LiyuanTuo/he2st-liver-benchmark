"""Build formal-report tables and figures from frozen result files."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'report'; FIG=OUT/'figures'; TABLE=OUT/'tables'


def escape(s):
    return str(s).replace('&',r'\&').replace('_',r'\_').replace('%',r'\%')


def table(path, headers, rows, align):
    lines=[r'\begin{tabular}{'+align+'}',r'\toprule',' & '.join(headers)+r' \\',r'\midrule']
    lines += [' & '.join(map(escape,row))+r' \\' for row in rows]
    lines += [r'\bottomrule',r'\end{tabular}']
    path.write_text('\n'.join(lines),encoding='utf8')


def main():
    FIG.mkdir(parents=True,exist_ok=True); TABLE.mkdir(exist_ok=True)
    final=json.loads((ROOT/'benchmarks/liver/final_ensemble.json').read_text())
    baseline=pd.read_csv(ROOT/'benchmarks/liver/baseline_20260913.csv')
    current=pd.read_csv(ROOT/'benchmarks/liver/optimization_20260914.csv')
    labels={'resnet18_ema':'新 ResNet18','efficientnet_b0_ema':'新 EfficientNet-B0','resnet18_spatial':'训练标签平滑 ResNet18','new_equal_ensemble':'三个新模型等权','previous_contextfusion':'原三种子集成','new_weight_0.25':'最终集成（新组25%）','new_weight_0.5':'新组50%','new_weight_0.75':'新组75%'}
    rows=[[r.method,*[f'{r[k]:.4f}' for k in ['HEG200','HEG50','MAE']]] for _,r in baseline.iterrows()]
    b=current[current.selected_on_C1].iloc[0]
    rows.append(['最终六模型集成',f'{b.D1_HEG200:.4f}',f'{b.D1_HEG50:.4f}',f'{b.D1_MAE:.4f}'])
    table(TABLE/'baseline.tex',['方法','HEG200 PCC','HEG50 PCC','MAE'],rows,'lrrr')
    rows=[[labels[r.candidate],*[f'{r[k]:.4f}' for k in ['C1_HEG200','C1_HEG50','D1_HEG200','D1_HEG50','D1_MAE']]] for _,r in current.iterrows()]
    table(TABLE/'optimization.tex',['候选','C1/200','C1/50','D1/200','D1/50','D1 MAE'],rows,'lrrrrr')
    rows=[]
    for c in final['components']:
        name={'old_seed42':'原 ResNet18 / seed42','old_seed17':'原 ResNet18 / seed17','old_seed83':'原 ResNet18 / seed83'}.get(c['name'],labels.get(c['name']))
        rows.append([name,str(c['epoch']),'四旋转' if c['tta'] else '单次','1/4' if c['name'].startswith('old_') else '1/12'])
    table(TABLE/'components.tex',['组成模型','选择轮次','推理增强','最终权重'],rows,'lrrr')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(10,4.8),layout='constrained');ax.set_xlim(0,10);ax.set_ylim(0,5);ax.axis('off')
    def box(x,y,w,h,text,color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.06',fc=color,ec='#7a8591'))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=10)
    def arrow(a,b):ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color='#475569',lw=1.5))
    box(.1,1.7,1.8,1.5,'H&E + spot XY\n224 / 896 / 1792 px\nresize to 224 x 224','#edf3f8')
    box(2.6,3.05,3.0,1.3,'Prior group: ResNet18\nseeds 42 / 17 / 83\nequal mean: O','#edf5ef')
    box(2.6,.55,3.0,1.6,'New group: R18 / B0 / R18-smooth\nseed 42, raw weights selected\nequal mean: N','#f7f4ef')
    box(6.5,1.65,3.1,1.6,'Final: 0.75 O + 0.25 N\nweight chosen on C1 HEG200\nD1: 0.2343 / 0.3180','#f9edec')
    arrow((1.9,2.7),(2.6,3.6));arrow((1.9,2.0),(2.6,1.35));arrow((5.6,3.6),(6.5,2.8));arrow((5.6,1.35),(6.5,2.05))
    ax.text(5,.05,'All averages operate in the same predicted panel-log expression space; no test RNA is used.',ha='center',fontsize=9)
    fig.savefig(FIG/'ensemble_method.pdf');fig.savefig(FIG/'ensemble_method.png',dpi=180);plt.close(fig)
    methods=['ST-Net (DenseNet121)','ContextFusion-scratch','ContextFusion-multiseed']
    selected=baseline.set_index('method').loc[methods]
    vals=np.vstack([selected[['HEG200','HEG50']].values,[b.D1_HEG200,b.D1_HEG50]])
    fig,ax=plt.subplots(figsize=(8,3.8),layout='constrained');x=np.arange(4)
    ax.bar(x-.18,vals[:,0],.36,label='HEG200');ax.bar(x+.18,vals[:,1],.36,label='HEG50')
    ax.set_xticks(x,['ST-Net','Scratch single','Prior 3-seed','Final 6-model']);ax.set_ylabel('D1 macro gene PCC');ax.set_ylim(0,.56)
    ax.axhline(.5,color='gray',ls='--',lw=.8);ax.legend(loc='upper left')
    for i in range(4):
        for j in range(2):ax.text(i+(-.18 if j==0 else .18),vals[i,j]+.007,f'{vals[i,j]:.4f}',ha='center',fontsize=8)
    fig.savefig(FIG/'comparison.pdf');fig.savefig(FIG/'comparison.png',dpi=180);plt.close(fig)
    d=dict(np.load(ROOT/'results/optimization_20260914/test_candidates.npz'))
    old=dict(np.load(ROOT/'results/liver_context_20260913/combined_predictions.npz'))
    for key in ['genes','barcodes','coordinates_xy','truth']:
        np.testing.assert_array_equal(old[key],d[key],err_msg=f'Misaligned spatial plot {key}')
    xy=d['coordinates_xy'];genes=d['genes'];chosen=d['new_weight_0.25']
    from scipy.stats import pearsonr
    fig,axes=plt.subplots(2,4,figsize=(11,5),layout='constrained')
    for row,gene in enumerate(['ALB','HP']):
        j=list(genes).index(gene);truth=d['truth'][:,j];lo,hi=np.quantile(truth,[.01,.99])
        for col,(name,value) in enumerate([('Measured',truth),('ST-Net',old['ST-Net (DenseNet121)'][:,j]),('Prior ensemble',d['previous_contextfusion'][:,j]),('Final ensemble',chosen[:,j])]):
            ax=axes[row,col];im=ax.scatter(xy[:,0],xy[:,1],c=value,s=4,cmap='viridis',vmin=lo,vmax=hi,rasterized=True)
            title=gene+' | '+name
            if col:title+=f'\nr={pearsonr(truth.astype(float),value.astype(float)).statistic:.3f}'
            ax.set_title(title,fontsize=9);ax.set_aspect('equal');ax.invert_yaxis();ax.set_axis_off()
        fig.colorbar(im,ax=axes[row,:],shrink=.8,label='Panel-relative log1p')
    fig.savefig(FIG/'spatial_examples.pdf');fig.savefig(FIG/'spatial_examples.png',dpi=180);plt.close(fig)
    # Error diversity is descriptive; it does not choose weights on D1.
    base_arrays=[]
    for c in final['components']:
        if c['name'].startswith('old_'):
            path=ROOT/c['checkpoint'];p=np.load(path.with_name('imagenet_D1.npz'))['predicted']
        else:p=d[c['name']]
        base_arrays.append(p)
    errors=np.stack([(p-d['truth']).ravel() for p in base_arrays])
    correlation=np.corrcoef(errors)
    pd.DataFrame(correlation,index=[c['name'] for c in final['components']],columns=[c['name'] for c in final['components']]).to_csv(ROOT/'benchmarks/liver/ensemble_residual_correlation.csv')
    mse={c['name']:float(np.square(p-d['truth']).mean()) for c,p in zip(final['components'],base_arrays)}
    mse['prior_ensemble']=float(np.square(d['previous_contextfusion']-d['truth']).mean())
    mse['final_ensemble']=float(np.square(chosen-d['truth']).mean())
    stats={'description':'Post-hoc error diversity on fixed D1; not used for model or weight selection','component_MSE':mse,'off_diagonal_error_correlation_min':float(correlation[np.triu_indices(6,1)].min()),'off_diagonal_error_correlation_max':float(correlation[np.triu_indices(6,1)].max())}
    (ROOT/'benchmarks/liver/ensemble_error_analysis.json').write_text(json.dumps(stats,indent=2),encoding='utf8')
    print('Built report tables and figures; error correlations:',stats['off_diagonal_error_correlation_min'],stats['off_diagonal_error_correlation_max'])


if __name__=='__main__':main()
