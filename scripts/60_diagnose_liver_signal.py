"""Label-side diagnostics, never used as H&E predictions or a claimed hard ceiling."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/liver_context_20260913';OUT.mkdir(parents=True,exist_ok=True)
def norm(x):return np.log1p(1e4*x/np.maximum(x.sum(1,keepdims=True),1e-12))
def pcc(x,y):
    x=x-x.mean(0);y=y-y.mean(0);den=np.sqrt((x*x).sum(0)*(y*y).sum(0))
    return np.divide((x*y).sum(0),den,out=np.full(x.shape[1],np.nan),where=den>1e-12)
def main():
    audit=pd.read_csv(ROOT/'results/verified_20260913/fixed_HEG200_gene_audit.csv')
    mask=audit.is_primary_HEG50.to_numpy(bool);rows=[];summary={}
    for slide in ['A1','B1','C1']:
        d=dict(np.load(ROOT/f'data/processed/gse240429_heg/arrays/C73_{slide}.npz'))
        raw=d['raw_counts'];y=norm(raw);ids=NearestNeighbors(n_neighbors=17).fit(d['coordinates_xy']).kneighbors(d['coordinates_xy'],return_distance=False)[:,1:]
        centered=y-y.mean(0);moran=(centered*centered[ids[:,:6]].mean(1)).sum(0)/(centered*centered).sum(0)
        # Binomial splitting conditions on observed counts. This is a sequencing
        # repeatability diagnostic, not independence of true biological replicates.
        split=[];rng=np.random.default_rng(42)
        for seed in range(20):
            a=rng.binomial(raw,.5);b=raw-a;split.append(pcc(norm(a),norm(b)))
        repeat=np.nanmean(split,axis=0)
        stats=dict(gene=d['genes'],HEG50=mask,mean_count=raw.mean(0),mean_expression=y.mean(0),std_expression=y.std(0),
                   moran6=moran,neighbor6_label_oracle=pcc(y,y[ids[:,:6]].mean(1)),
                   neighbor16_label_oracle=pcc(y,y[ids].mean(1)),half_count_repeatability=repeat)
        frame=pd.DataFrame(stats);frame.insert(0,'slide',slide);rows.append(frame)
        summary[slide]={key:dict(HEG200=float(np.nanmean(stats[key])),HEG50=float(np.nanmean(np.asarray(stats[key])[mask]))) for key in ['moran6','neighbor6_label_oracle','neighbor16_label_oracle','half_count_repeatability']}
    pd.concat(rows).to_csv(OUT/'train_validation_signal_diagnostics.csv',index=False)
    summary['interpretation']={'oracle':'Uses measured neighbor RNA within the same slide, so cannot be an H&E benchmark entry or deployment model.',
       'repeatability':'Binomial read splitting at half depth; conditional on observed counts, not true technical replicates. Not a rigorous attainable-PCC upper bound.',
       'moran':'Abundance and spatial organization are different. Moran weights are uniform 6 nearest OTHER spots.',
       'test_labels_used':False}
    (OUT/'signal_diagnostics.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
