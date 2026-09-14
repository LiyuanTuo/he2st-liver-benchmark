"""Post-hoc C1 count-thinning diagnostic; never alters the D1 benchmark.

Captured UMI differences can reflect biology, capture, and sequencing together.
Thinning is a proxy experiment, not proof of a purely technical depth cause.
"""
from pathlib import Path
import json
import anndata as ad
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/liver_context_20260913'
def norm(x):return np.log1p(1e4*x/np.maximum(x.sum(1,keepdims=True),1e-12))
def corr(x,y):
    x=x-x.mean(0);y=y-y.mean(0);return (x*y).sum(0)/np.sqrt((x*x).sum(0)*(y*y).sum(0))
def main():
    depth=json.loads((OUT/'count_depth_diagnostics.json').read_text());medians={}
    for s in ['A1','B1','C1','D1']:
        a=ad.read_h5ad(ROOT/f'data/processed/gse240429_heg/st/C73_{s}.h5ad')
        medians[s]=float(np.median(a.obs['total_counts_all_genes']))
        np.testing.assert_allclose(medians[s],depth['slides'][s]['median_total_all_genes'],atol=.001)
    depth['exact_whole_library_verified']=True;depth['exact_medians']=medians
    (OUT/'count_depth_diagnostics.json').write_text(json.dumps(depth,indent=2))
    ratio=medians['D1']/medians['C1'];raw=np.load(ROOT/'data/processed/gse240429_heg/arrays/C73_C1.npz')['raw_counts']
    import pandas as pd
    mask=pd.read_csv(ROOT/'results/verified_20260913/fixed_HEG200_gene_audit.csv').is_primary_HEG50.to_numpy(bool)
    predictions={arm:np.load(ROOT/f'results/liver_context_fusion_20260913/{arm}_C1.npz')['predicted'].astype(float) for arm in ['imagenet','scratch']}
    values={arm:[] for arm in predictions};rng=np.random.default_rng(123)
    for repeat in range(30):
        y=norm(rng.binomial(raw,ratio))
        for arm,p in predictions.items():
            r=corr(y,p);values[arm].append([r.mean(),r[mask].mean()])
    summary=dict(UMI_medians=medians,keep_probability=ratio,repeats=30,
      unchanged='All predictions and D1 benchmark truth unchanged. Only a diagnostic copy of C1 counts is thinned.',
      limitation='Captured-UMI differences include biological RNA content, capture efficiency and sequencing. Binomial thinning is a proxy, not identification of a purely technical cause.',models={})
    for arm,p in predictions.items():
        r=corr(norm(raw),p);v=np.array(values[arm])
        summary['models'][arm]=dict(original_C1=[float(r.mean()),float(r[mask].mean())],thinned_C1_mean=v.mean(0).tolist(),thinned_C1_std=v.std(0,ddof=1).tolist())
    (OUT/'count_thinning_ablation.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
