"""One fixed HEG panel, one target scale, no test-based gene/model selection.

Main metric: per-gene correlation across spots, macro over train-selected HEG50
and HEG200, against unprojected panel-relative log expression. PCA is secondary.
Run after scripts 40/43, 39/42 and HE2ST_PANEL=gse240429_heg script 29.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/verified_20260913'
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / 'src'))
from he2st.metrics import evaluate_normalized, per_gene_pearson

METHODS = [
    ('Image Ridge', 'unified_image_baselines_heg/image_ridge_C73_D1.npz', 'predicted', 'log'),
    ('MLP', 'unified_image_baselines_heg/stnet_style_mlp_C73_D1.npz', 'predicted', 'log'),
    ('BLEEP', 'improvement_heg/bleep/C73_D1_predictions.npz', 'predicted', 'log'),
    ('ResSAT', 'ressat_unified_heg/unified_batch16_predictions.npz', 'predicted', 'log'),
    ('GenAR', 'improvement_heg/genar/C73_D1_predictions.npz', 'predicted_counts', 'count'),
    ('Stem', 'improvement_heg/stem/C73_D1_predictions.npz', 'predicted', 'log'),
    ('ST-Net (DenseNet121)', 'improvement_heg/stnet/C73_D1_predictions.npz', 'predicted', 'log'),
]


def norm(values):
    values = np.maximum(np.asarray(values, dtype=np.float64), 0)
    return np.log1p(10000 * values / np.maximum(values.sum(1, keepdims=True), 1e-12))


def clean(obj):
    if isinstance(obj, dict): return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, list): return [clean(v) for v in obj]
    if isinstance(obj, np.generic): obj = obj.item()
    if isinstance(obj, float) and not np.isfinite(obj): return None
    return obj


def load_aligned(path, key, kind, data):
    with np.load(path) as d:
        gene_key = 'genes' if 'genes' in d else 'gene_names'
        np.testing.assert_array_equal(d[gene_key], data['genes'])
        if 'coordinates_xy' in d:
            np.testing.assert_array_equal(d['coordinates_xy'], data['coordinates_xy'])
        if 'true_counts' in d:
            np.testing.assert_array_equal(d['true_counts'], data['raw_counts'])
        elif 'true' in d:
            np.testing.assert_allclose(d['true'], data['log_normalized'], atol=1e-6)
        else:
            raise ValueError(f'Missing row-alignment evidence: {path}')
        pred = d[key].astype(float)
    assert pred.shape == data['raw_counts'].shape and np.isfinite(pred).all(), path
    abundance = pred if kind == 'count' else np.expm1(pred)
    return norm(abundance), pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--allow-pending', action='store_true')
    args = ap.parse_args()
    base = ROOT / 'data/processed/gse240429_heg/arrays'
    data = dict(np.load(base / 'C73_D1.npz'))
    train = [dict(np.load(base / f'C73_{s}.npz')) for s in ['A1', 'B1']]
    audit = pd.read_csv(OUT / 'fixed_HEG200_gene_audit.csv')
    np.testing.assert_array_equal(audit.gene, data['genes'])
    mask = audit.is_primary_HEG50.to_numpy(bool)
    assert mask.sum() == 50
    truth = norm(data['raw_counts'])
    train_panel = np.concatenate([norm(d['raw_counts']) for d in train])
    center, scale = train_panel.mean(0), np.maximum(train_panel.std(0), 1e-8)
    pca = PCA(50, svd_solver='full').fit((train_panel - center) / scale)
    def recon(a):
        return pca.inverse_transform(pca.transform((a - center) / scale)) * scale + center
    reconstructed_truth = recon(truth)
    summary = {'protocol': {
        'dataset': 'GSE240429, one donor C73, train A1+B1 / validation C1 / test D1',
        'gene_selection': 'Top 200 by mean whole-transcriptome log1p-normalized expression in training only',
        'HEG50_selection': 'First 50 in exactly the same training-only global expression ranking; fixed before predictions',
        'main_target': 'log1p(10000 * nonnegative abundance / total abundance of fixed HEG200 panel)',
        'main_PCC': 'Per gene across 2265 spots, macro over all HEG200 and fixed HEG50; no PCA applied to truth',
        'secondary_PCA': '50 components fit only on A1+B1; diagnostic, never a replacement for main PCC',
        'limitation': 'Common task/input patches/split/genes/evaluator, but method-specific encoders and compute; one donor, one test slide; no general method ranking',
        'reproducibility': 'Every prediction checked for exact gene order and matching truth and/or coordinates; source hashes recorded',
    }, 'methods': {}, 'pending': [], 'same_target_ablation': {}}
    all_pcc = pd.DataFrame({'gene': data['genes'], 'training_HEG50': mask})
    prediction_export = dict(truth=truth.astype(np.float32), genes=data['genes'], barcodes=data['barcodes'], coordinates_xy=data['coordinates_xy'])
    for name, file, key, kind in METHODS:
        path = ROOT / 'results' / file
        if not path.exists():
            summary['pending'].append(name)
            continue
        pred, native = load_aligned(path, key, kind, data)
        metrics, pcc = evaluate_normalized(truth, pred, data['coordinates_xy'])
        pc = per_gene_pearson(reconstructed_truth, recon(pred))
        with open(path, 'rb') as f:
            digest = hashlib.file_digest(f, 'sha256').hexdigest()
        row = dict(HEG200_PCC=metrics['pcc_macro_all_fixed_genes'], HEG50_PCC=float(np.nanmean(pcc[mask])),
                   defined_HEG200=int(np.isfinite(pcc).sum()), defined_HEG50=int(np.isfinite(pcc[mask]).sum()),
                   HEG200_Spearman=metrics['spearman_macro_all_fixed_genes'], MAE=metrics['mae_log_normalized'],
                   RMSE=metrics['rmse_log_normalized'], Moran_I_MAE=metrics['moran_i_mae'],
                   PCA_diagnostic_HEG200=float(np.nanmean(pc)), PCA_diagnostic_HEG50=float(np.nanmean(pc[mask])),
                   source=str(path.relative_to(ROOT)), sha256=digest, gene_and_spot_alignment_passed=True)
        if kind == 'log':
            direct = per_gene_pearson(data['log_normalized'], native)
            row['whole_transcriptome_scale_PCC_HEG200_diagnostic'] = float(np.nanmean(direct))
            row['whole_transcriptome_scale_PCC_HEG50_diagnostic'] = float(np.nanmean(direct[mask]))
        summary['methods'][name] = row
        all_pcc[name] = pcc
        prediction_export[name] = pred.astype(np.float32)
        print(name, row['HEG200_PCC'], row['HEG50_PCC'], flush=True)
    # Hold gene set/true scale fixed to distinguish decoder changes from panel changes.
    for name, file, key, kind in [
        ('GenAR top1, same forward pass', 'improvement_heg/genar/C73_D1_top1_control.npz', 'predicted_counts', 'count'),
        ('BLEEP original head', 'unified_image_baselines_heg/bleep_C73_D1.npz', 'predicted', 'log')]:
        path = ROOT / 'results' / file
        if path.exists():
            pred, native = load_aligned(path, key, kind, data)
            pc = per_gene_pearson(truth, pred)
            summary['same_target_ablation'][name] = dict(HEG200_PCC=float(np.nanmean(pc)), HEG50_PCC=float(np.nanmean(pc[mask])))
    if summary['pending'] and not args.allow_pending:
        raise RuntimeError(f'Incomplete benchmark: {summary["pending"]}')
    stem = 'benchmark_partial' if summary['pending'] else 'benchmark'
    (OUT / f'{stem}.json').write_text(json.dumps(clean(summary), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
    pd.DataFrame(summary['methods']).T.to_csv(OUT / f'{stem}.csv')
    all_pcc.to_csv(OUT / f'{stem}_per_gene.csv', index=False)
    np.savez_compressed(OUT / f'{stem}_predictions.npz', **prediction_export)
    print('Saved', stem, 'pending:', summary['pending'])


if __name__ == '__main__':
    main()
