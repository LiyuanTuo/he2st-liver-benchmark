"""Run the declared 3-config sweep, lock on C1, then evaluate D1 once.

An optional previous model prediction is an explicitly supplied control.
Every candidate is retained, including candidates that fail to improve.
"""
from pathlib import Path
import argparse
import json
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from he2st.engine import fit, predict, scores, write_json
from he2st.data import panel_log1p, read_panel

CONFIGS = ['resnet18_ema', 'efficientnet_b0_ema', 'resnet18_spatial']


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=ROOT/'results/optimization_20260914')
    ap.add_argument('--baseline-validation', type=Path)
    ap.add_argument('--baseline-test', type=Path)
    args = ap.parse_args()
    if bool(args.baseline_validation) != bool(args.baseline_test):
        ap.error('Supply both baseline files or neither')
    output = args.output; output.mkdir(parents=True, exist_ok=True)
    plans = {name: json.loads((ROOT/f'configs/{name}.json').read_text()) for name in CONFIGS}
    write_json(output/'plan.json', {'configs': plans, 'selection': 'C1 HEG200 PCC only',
        'ensemble_candidates': 'Each model; equal 3-model mean; best-new/previous mixtures 0.25, 0.5, 0.75 if previous supplied',
        'test_truth_unchanged': True, 'started_unix': time.time()})
    for name in CONFIGS:
        fit(ROOT, plans[name], output/name)
    genes, mask = read_panel(ROOT/'benchmarks/liver/genes.csv')
    val_truth = dict(np.load(ROOT/plans[CONFIGS[0]]['arrays']/'C73_C1.npz'))
    yv = panel_log1p(val_truth['raw_counts'])
    vp = {}
    for name in CONFIGS:
        with np.load(output/name/'validation.npz') as d:
            for key in ['genes', 'barcodes', 'coordinates_xy']:
                np.testing.assert_array_equal(d[key], val_truth[key])
            np.testing.assert_array_equal(d['truth'], yv)
            vp[name] = d['predicted']
    vp['new_equal_ensemble'] = np.mean([vp[name] for name in CONFIGS], axis=0)
    best_new = max(vp, key=lambda name: scores(yv, vp[name], mask)['HEG200'])
    if args.baseline_validation:
        with np.load(args.baseline_validation) as baseline:
            np.testing.assert_array_equal(baseline['genes'], genes)
            np.testing.assert_array_equal(baseline['truth'], yv)
            vp['previous_contextfusion'] = baseline['predicted']
        for weight in [.25, .5, .75]:
            vp[f'new_weight_{weight}'] = weight*vp[best_new] + (1-weight)*vp['previous_contextfusion']
    validation = {name: scores(yv, p, mask) for name, p in vp.items()}
    selected = max(validation, key=lambda name: validation[name]['HEG200'])
    selection = {'selected': selected, 'best_new': best_new, 'validation': validation, 'locked_unix': time.time()}
    write_json(output/'selection_locked.json', selection)
    np.savez_compressed(output/'validation_candidates.npz', truth=yv, genes=genes, **vp)
    # First access to D1 RNA occurs after all fitting and ensemble selection.
    td = dict(np.load(ROOT/plans[CONFIGS[0]]['arrays']/'C73_D1.npz'))
    yt = panel_log1p(td['raw_counts']); tp = {}
    for name in CONFIGS:
        path = predict(ROOT, output/name/'best.pt', ['C73_D1'], output/name/'D1_prediction.npz')
        with np.load(path) as pred:
            for key in ['genes', 'barcodes', 'coordinates_xy']:
                np.testing.assert_array_equal(pred[key], td[key])
            tp[name] = pred['predicted']
    tp['new_equal_ensemble'] = np.mean([tp[name] for name in CONFIGS], axis=0)
    if args.baseline_test:
        with np.load(args.baseline_test) as baseline:
            for key in ['genes', 'barcodes', 'coordinates_xy']:
                np.testing.assert_array_equal(baseline[key], td[key])
            np.testing.assert_array_equal(baseline['truth'], yt)
            tp['previous_contextfusion'] = baseline['predicted']
        for weight in [.25, .5, .75]:
            tp[f'new_weight_{weight}'] = weight*tp[best_new] + (1-weight)*tp['previous_contextfusion']
    test = {name: scores(yt, p, mask) for name, p in tp.items()}
    write_json(output/'test_metrics.json', {'selected': selected, 'test': test})
    np.savez_compressed(output/'test_candidates.npz', truth=yt, genes=genes, barcodes=td['barcodes'],
        coordinates_xy=td['coordinates_xy'], heg50_mask=mask, **tp)
    np.savez_compressed(output/'selected_D1.npz', predicted=tp[selected], truth=yt, genes=genes,
        barcodes=td['barcodes'], coordinates_xy=td['coordinates_xy'], heg50_mask=mask)
    print('SELECTED', selected, test[selected], flush=True)


if __name__ == '__main__':
    main()
