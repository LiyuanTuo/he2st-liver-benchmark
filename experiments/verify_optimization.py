"""Reinfer saved models, verify fixed-panel scores, and estimate paired uncertainty."""
from pathlib import Path
import json
import sys

import numpy as np
from scipy.stats import pearsonr

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from he2st.engine import predict, scores, write_json
from he2st.metrics import per_gene_pearson


def main():
    output=ROOT/'results/optimization_20260914'
    selected=json.loads((output/'selection_locked.json').read_text())['selected']
    report=json.loads((output/'test_metrics.json').read_text())['test']
    d=dict(np.load(output/'test_candidates.npz'))
    y,mask=d['truth'],d['heg50_mask']
    check={}
    for name in ['resnet18_ema','efficientnet_b0_ema','resnet18_spatial']:
        fresh=output/name/'D1_fresh.npz'
        if not fresh.exists():
            predict(ROOT,output/name/'best.pt',['C73_D1'],fresh)
        with np.load(fresh) as pred:
            for key in ['genes','barcodes','coordinates_xy']:
                np.testing.assert_array_equal(pred[key],d[key])
            np.testing.assert_allclose(pred['predicted'],d[name],rtol=0,atol=1e-5)
            check[name]={'max_absolute_difference':float(np.abs(pred['predicted']-d[name]).max()),'fresh_checkpoint_reinference':True}
    for name,expected in report.items():
        actual=scores(y,d[name],mask)
        for key in ['HEG200','HEG50','MAE']:
            np.testing.assert_allclose(actual[key],expected[key],rtol=0,atol=1e-12)
        r=per_gene_pearson(y,d[name])
        for j in [0,25,100,199]:
            np.testing.assert_allclose(r[j],pearsonr(y[:,j].astype(float),d[name][:,j].astype(float)).statistic,atol=1e-12)
    # Same spatial-block rule as the previous round, with paired resampling.
    xy=d['coordinates_xy']
    bins=np.minimum(((xy-xy.min(0))/(np.ptp(xy,axis=0)+1e-9)*5).astype(int),4)
    ids=bins[:,0]+5*bins[:,1]
    groups=[np.flatnonzero(ids==i) for i in np.unique(ids)]
    rng=np.random.default_rng(42); sampled=[]
    for _ in range(300):
        rows=np.concatenate([groups[i] for i in rng.integers(0,len(groups),len(groups))])
        a=scores(y[rows],d[selected][rows],mask)
        b=scores(y[rows],d['previous_contextfusion'][rows],mask)
        sampled.append([a['HEG200'],a['HEG50'],a['HEG200']-b['HEG200'],a['HEG50']-b['HEG50']])
    ci=np.quantile(sampled,[.025,.975],axis=0)
    result={'selected':selected,'fresh_predictions':check,'all_candidates_recomputed':True,
            'spatial_blocks':len(groups),'bootstrap_repeats':300,
            'CI95':dict(zip(['HEG200','HEG50','delta_HEG200_vs_previous','delta_HEG50_vs_previous'],ci.T.tolist())),
            'limitation':'Conditional on this one slide; does not quantify independent-patient uncertainty or repeated development bias.'}
    write_json(output/'verification.json',result)
    write_json(ROOT/'benchmarks/liver/optimization_verification.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
