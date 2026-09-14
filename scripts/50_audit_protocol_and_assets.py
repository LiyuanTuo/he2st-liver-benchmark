"""Recheck source hashes, training-only HEG ranks, and saved ResSAT predictions.

Run with the existing global WSL python3. No training and no test-based tuning.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/verified_20260913'
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / 'src'))
from he2st.metrics import per_gene_pearson, per_gene_morans_i


def sha(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf8')


def main():
    # Independent reference and edge cases: constants must not get fake PCCs.
    rng = np.random.default_rng(42)
    a, b = rng.normal(size=(100, 5)), rng.normal(size=(100, 5))
    np.testing.assert_allclose(per_gene_pearson(a, b), [pearsonr(a[:, i], b[:, i]).statistic for i in range(5)], atol=1e-12)
    assert np.isnan(per_gene_pearson(a, np.full_like(a, .1))).all()
    np.testing.assert_allclose(per_gene_pearson(a, a), 1)
    np.testing.assert_allclose(per_gene_pearson(a, -a), -1)
    # Explicit nearest-neighbor result: 0->1, 1->0, 2->1.
    np.testing.assert_allclose(per_gene_morans_i(np.array([[0.], [1.], [3.]]), np.array([[0, 0], [1, 0], [3, 0]]), 1), 1/14)

    integrity = []
    base = ROOT / 'data/raw/ressat_mouse_brain'
    manifest = json.loads((base / 'manifest.json').read_text())
    for sample, section in manifest['samples'].items():
        for entry in section['files']:
            p = base / sample / entry['name']
            digest = sha(p)
            item = dict(path=str(p.relative_to(ROOT)), bytes=p.stat().st_size,
                        sha256=digest, matches_manifest=digest == entry['sha256'] and p.stat().st_size == entry['bytes'])
            integrity.append(item)
            assert item['matches_manifest'], item
    base = ROOT / 'data/raw/hest_bench/IDC'
    for entry in json.loads((base / 'manifest.json').read_text())['files']:
        p = base / entry['path']
        digest = sha(p)
        item = dict(path=str(p.relative_to(ROOT)), bytes=p.stat().st_size, sha256=digest,
                    matches_manifest=digest == entry['sha256'] and p.stat().st_size == entry['bytes'])
        integrity.append(item)
        assert item['matches_manifest'], item
    write('source_integrity.json', integrity)
    print('Manifest hashes verified:', len(integrity), flush=True)

    spec = importlib.util.spec_from_file_location('heg_builder', ROOT / 'scripts/34_build_heg_panel.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    features = m.load_feature_table()
    train_parts = [m.load_slide(s)[0] for s in ['1', '2']]
    train = sparse.vstack(train_parts, format='csr')
    means = np.asarray(m.log_normalize(train).mean(0)).ravel()
    selected, names, diagnostics = m.select_heg(train, features)
    with np.load(ROOT / 'data/processed/gse240429_heg/arrays/C73_A1.npz') as d:
        saved = d['genes'].tolist()
    assert set(names) == set(saved)
    full_top50 = names[:50]
    with np.load(ROOT / 'data/processed/gse240429/arrays/C73_A1.npz') as d:
        old_names = d['genes'].tolist()
    ranks = {name: i + 1 for i, name in enumerate(names)}
    gene_idx = dict(zip(features.gene_name, range(len(features))))
    train_counts = train[:, [gene_idx[g] for g in saved]].toarray()
    gene_table = pd.DataFrame({'gene': saved, 'train_expression_rank': [ranks[g] for g in saved],
                              'train_mean_log1p_all_genes': [float(means[gene_idx[g]]) for g in saved],
                              'train_detection_fraction': (train_counts > 0).mean(0),
                              'is_primary_HEG50': [g in full_top50 for g in saved]})
    gene_table.to_csv(OUT / 'fixed_HEG200_gene_audit.csv', index=False)
    report = {'selection': 'Top 200 mean log1p(10000 * count / whole-transcriptome library), A1+B1 only',
              'primary_top50': full_top50, 'old_panel_overlap_with_full_top50': len(set(old_names) & set(full_top50)),
              'old_panel_overlap_with_HEG200': len(set(old_names) & set(names)),
              'diagnostics': diagnostics, 'slides': {}, 'fixed_genes_sha256': m.sha256_text(saved)}
    for slide, number in m.SLIDES.items():
        counts, barcodes, xy = m.load_slide(number)
        with np.load(ROOT / f'data/processed/gse240429_heg/arrays/{slide}.npz') as new, np.load(ROOT / f'data/processed/gse240429/arrays/{slide}.npz') as old:
            np.testing.assert_array_equal(new['barcodes'], barcodes)
            np.testing.assert_array_equal(new['barcodes'], old['barcodes'])
            np.testing.assert_array_equal(new['coordinates_xy'], xy)
            np.testing.assert_array_equal(new['genes'], saved)
            np.testing.assert_array_equal(new['raw_counts'], counts[:, [gene_idx[g] for g in saved]].toarray())
            np.testing.assert_allclose(new['log_normalized'], m.log_normalize(counts)[:, [gene_idx[g] for g in saved]].toarray(), atol=1e-6)
            report['slides'][slide] = dict(spots=len(barcodes), HEG_zero_fraction=float((new['raw_counts'] == 0).mean()),
                                            old_zero_fraction=float((old['raw_counts'] == 0).mean()), barcode_and_gene_alignment=True)
    write('panel_audit.json', report)
    print('Panel independently reconstructed from raw matrix: PASS', report['old_panel_overlap_with_full_top50'], flush=True)

    paper = {}
    for name, file, key, reference, reference_heg in [
        ('SA', 'ressat_official/official_batch32_predictions.npz', 'true', .6577, .8781),
        ('SP', 'ressat_sp_rebuilt/SP1_predictions.npz', 'reconstructed_true', .6980, .8999)]:
        p = ROOT / 'results' / file
        with np.load(p) as d:
            truth, pred = d[key].astype(float), d['predicted'].astype(float)
            pcc = per_gene_pearson(truth, pred)
            heg = np.argsort(truth.mean(0))[-50:]
            paper[name] = dict(source=str(p.relative_to(ROOT)), sha256=sha(p),
                               paper_HVG2000=reference, reproduced_HVG2000=float(np.nanmean(pcc)),
                               paper_HEG50=reference_heg, reproduced_HEG50=float(np.nanmean(pcc[heg])),
                               HEG_selection='Top mean observed reconstructed expression within 2000 HVGs; paper diagnostic, test-selected',
                               spots=len(truth), genes=truth.shape[1])
            if 'strict_true' in d:
                strict = d['strict_true'].astype(float)
                strict_pcc = per_gene_pearson(strict, pred)
                paper[name]['strict_truth_HVG2000_defined_macro'] = float(np.nanmean(strict_pcc))
                paper[name]['strict_truth_defined_genes'] = int(np.isfinite(strict_pcc).sum())
                paper[name]['strict_truth_same_HEG50'] = float(np.nanmean(strict_pcc[heg]))
            pd.DataFrame({'gene': d['genes'], 'reconstructed_truth_pcc': pcc,
                          'paper_HEG50': np.isin(np.arange(len(pcc)), heg)}).to_csv(OUT / f'ResSAT_{name}_gene_pcc.csv', index=False)
    write('ressat_recomputed.json', paper)
    print(json.dumps(paper, indent=2), flush=True)


if __name__ == '__main__':
    main()
