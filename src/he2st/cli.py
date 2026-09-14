"""Small public command line: doctor, fit, predict, evaluate."""
from pathlib import Path
import argparse
import importlib.metadata
import json
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description='H&E → fixed-panel spatial gene expression')
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2], help='Project/data root')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('doctor', help='Check the existing Python environment; installs nothing')
    prep = commands.add_parser('prepare', help='Rebuild fixed arrays and image caches from local source files')
    prep.add_argument('--source', type=Path, required=True, help='BLEEP GSE240429_data/data directory')
    prep.add_argument('--tiffs', type=Path, default=Path('data/processed/gse240429/tiff'))
    prep.add_argument('--arrays', type=Path, default=Path('data/processed/gse240429_heg/arrays'))
    prep.add_argument('--images', type=Path, default=Path('data/processed/gse240429/context_20260913'))
    prep.add_argument('--panel', type=Path, default=Path('benchmarks/liver/genes.csv'))
    prep.add_argument('--arrays-only', action='store_true')
    fit = commands.add_parser('fit', help='Train and select using train/validation only')
    fit.add_argument('--config', type=Path, required=True)
    fit.add_argument('--output', type=Path, required=True)
    fit.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    pred = commands.add_parser('predict', help='Predict from prepared images; never loads RNA counts')
    pred.add_argument('--checkpoint', type=Path, required=True)
    pred.add_argument('--slides', nargs='+', required=True)
    pred.add_argument('--output', type=Path, required=True)
    pred.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    ens = commands.add_parser('ensemble', help='Predict the frozen six-model ensemble from images only')
    ens.add_argument('--manifest', type=Path, default=Path('benchmarks/liver/final_ensemble.json'))
    ens.add_argument('--slides', nargs='+', required=True)
    ens.add_argument('--output', type=Path, required=True)
    ens.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    ev = commands.add_parser('evaluate', help='Score saved predictions against unprojected panel truth')
    ev.add_argument('--prediction', type=Path, required=True)
    ev.add_argument('--truth', type=Path, required=True)
    ev.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'doctor':
        result = {'python': sys.version.split()[0], 'executable': sys.executable, 'packages': {}}
        for name in ['torch', 'torchvision', 'numpy', 'scipy', 'pandas', 'scikit-learn', 'Pillow', 'tifffile']:
            try:
                result['packages'][name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                result['packages'][name] = 'missing'
        try:
            import torch
            result['cuda_available'] = torch.cuda.is_available()
            if torch.cuda.is_available():
                result['gpu'] = torch.cuda.get_device_name(0)
        except ImportError:
            result['cuda_available'] = False
        print(json.dumps(result, indent=2))
        return
    if args.command == 'prepare':
        from .preprocessing import prepare_benchmark
        prepare_benchmark(args.root, args.root/args.source, args.root/args.tiffs,
                          args.arrays, args.images, args.panel, args.arrays_only)
        return
    if args.command == 'ensemble':
        from .ensemble import predict_ensemble
        print(predict_ensemble(args.root, args.manifest, args.slides, args.output, args.device))
        return
    from .engine import fit, predict, scores, write_json
    if args.command == 'fit':
        fit(args.root, json.loads(args.config.read_text(encoding='utf8')), args.output, args.device)
    elif args.command == 'predict':
        print(predict(args.root, args.checkpoint, args.slides, args.output, args.device))
    elif args.command == 'evaluate':
        import numpy as np
        from .data import panel_log1p, read_panel
        from .metrics import per_gene_pearson
        with np.load(args.prediction) as pred, np.load(args.truth) as true:
            for key in ['genes', 'barcodes', 'coordinates_xy']:
                np.testing.assert_array_equal(pred[key], true[key], err_msg=f'Misaligned {key}')
            target = panel_log1p(true['raw_counts'])
            mask = np.asarray(pred['heg50_mask'], dtype=bool)
            fixed_genes, fixed_mask = read_panel(args.root/'benchmarks/liver/genes.csv')
            np.testing.assert_array_equal(true['genes'], fixed_genes, err_msg='Changed benchmark panel')
            np.testing.assert_array_equal(mask, fixed_mask, err_msg='Changed training-selected HEG50 mask')
            result = scores(target, pred['predicted'], mask)
            result.update(spots=len(target), target='unprojected panel-relative log1p', gene_selection='training only')
            write_json(args.output, result)
            import pandas as pd
            pd.DataFrame({'gene': true['genes'], 'HEG50': mask, 'PCC': per_gene_pearson(target, pred['predicted'])}).to_csv(args.output.with_suffix('.per_gene.csv'), index=False)
            print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
