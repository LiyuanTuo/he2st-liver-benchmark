"""Bounded prediction smoothing control; lock C1 choice before reading D1.

Run once from frozen ensemble predictions. No model fitting, gene reselection,
or target smoothing. The historical six-model result remains the reference.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'benchmarks/liver/report_spatial_ablation'


def neighbors(xy, k):
    # Explicitly remove self even for coincident coordinates.
    index = cKDTree(xy).query(xy, k=k+1)[1]
    return np.stack([row[row != i][:k] for i, row in enumerate(index)])


def smooth(pred, xy, k, alpha):
    if alpha == 0:
        return pred.copy()
    return (1-alpha)*pred + alpha*pred[neighbors(xy, k)].mean(axis=1)


def metrics(pred, truth, mask):
    a = pred.astype('float64') - pred.mean(axis=0, dtype='float64')
    b = truth.astype('float64') - truth.mean(axis=0, dtype='float64')
    r = (a*b).sum(0)/np.sqrt((a*a).sum(0)*(b*b).sum(0))
    assert np.isfinite(r).all()
    return {'HEG200': float(r.mean()), 'HEG50': float(r[mask].mean()),
            'MAE': float(np.abs(pred-truth).mean())}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plan = {'reference': 'frozen new_weight_0.25', 'k': [6, 12, 24],
            'alpha': [0.25, 0.5, 0.75, 1.0], 'include_identity': True,
            'selection': 'maximum C1 HEG200; ties prefer identity then smaller k/alpha',
            'D1_role': 'evaluate identity and C1-selected candidate only',
            'target_smoothing': False, 'retraining': False}
    (OUT/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    base = ROOT/'results/optimization_20260914'
    with np.load(base/'validation_candidates.npz') as v, np.load(ROOT/'data/processed/gse240429_heg/arrays/C73_C1.npz') as meta:
        np.testing.assert_array_equal(v['genes'], meta['genes'])
        counts = meta['raw_counts']
        truth = np.log1p(1e4*counts/counts.sum(1, keepdims=True))
        np.testing.assert_allclose(v['truth'], truth, atol=1e-6, rtol=1e-6)
        genes = pd.read_csv(ROOT/'benchmarks/liver/genes.csv')
        np.testing.assert_array_equal(v['genes'], genes.gene.values)
        mask = genes.is_primary_HEG50.to_numpy(dtype=bool)
        pred = v['new_weight_0.25']; xy = meta['coordinates_xy']
        rows = [{'k': 0, 'alpha': 0., **metrics(pred, v['truth'], mask)}]
        for k in plan['k']:
            for alpha in plan['alpha']:
                rows.append({'k': k, 'alpha': alpha, **metrics(smooth(pred, xy, k, alpha), v['truth'], mask)})
    pd.DataFrame(rows).to_csv(OUT/'validation.csv', index=False)
    best = max(rows, key=lambda x: x['HEG200'])
    locked = {'selected': best, 'D1_loaded_at_selection': False,
              'validation_predictions_sha256': hashlib.sha256((base/'validation_candidates.npz').read_bytes()).hexdigest()}
    (OUT/'selection_locked.json').write_text(json.dumps(locked, indent=2)+'\n')
    # First access to the fixed test predictions in this experiment is below.
    with np.load(base/'test_candidates.npz') as d:
        np.testing.assert_array_equal(d['genes'], genes.gene.values)
        y = d['truth']; p = d['new_weight_0.25']
        q = smooth(p, d['coordinates_xy'], best['k'], best['alpha'])
        result = {'reference_D1': metrics(p, y, mask), 'selected_D1': metrics(q, y, mask),
                  'selected_k': best['k'], 'selected_alpha': best['alpha'],
                  'interpretation': 'Supplemental C1-selected control; no D1 search; historical main result preserved'}
    (OUT/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
