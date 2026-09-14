"""Rebuild the fixed benchmark from BLEEP counts and full-resolution RGB TIFFs."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread
from PIL import Image
import tifffile

from .data import read_panel, gene_digest

SLIDES = {'C73_A1': '1', 'C73_B1': '2', 'C73_C1': '3', 'C73_D1': '4'}


def read_source(source, number):
    folder = Path(source)/'filtered_expression_matrices'/number
    features = pd.read_csv(folder/'features.tsv', sep='\t', header=None).fillna('')
    counts = mmread(folder/'matrix.mtx').tocsr().T.tocsr().astype(np.int32)
    barcodes = pd.read_csv(folder/'barcodes.tsv', sep='\t', header=None)[0].to_numpy(str)
    positions = pd.read_csv(Path(source)/'tissue_pos_matrices'/f'tissue_positions_list_{number}.csv',
                            header=None, names=['barcode', 'in_tissue', 'row', 'col', 'y', 'x']).set_index('barcode')
    if len(set(barcodes)) != len(barcodes) or not positions.index.is_unique:
        raise ValueError('Duplicate barcodes in source')
    positions = positions.loc[barcodes]
    if counts.shape != (len(barcodes), len(features)) or not (positions.in_tissue == 1).all():
        raise ValueError('Count dimensions or tissue membership mismatch')
    if (counts.data < 0).any():
        raise ValueError('Negative counts')
    return counts, features, barcodes, positions[['x', 'y']].to_numpy(np.float32)


def whole_log1p(counts):
    total = np.asarray(counts.sum(1)).ravel().astype(np.float64)
    if (total <= 0).any():
        raise ValueError('Empty expression row')
    values = counts.astype(np.float64).multiply((10000/total)[:, None]).tocsr()
    values.data = np.log1p(values.data)
    return values.astype(np.float32)


def crop_view(image, x, y, size):
    if size <= 0 or size % 2:
        raise ValueError('View size must be positive and even')
    x, y = int(round(float(x))), int(round(float(y)))
    left, top = x-size//2, y-size//2
    x0, x1 = max(0, left), min(image.shape[1], left+size)
    y0, y1 = max(0, top), min(image.shape[0], top+size)
    if x0 >= x1 or y0 >= y1:
        raise ValueError('Spot is outside the image')
    patch = np.full((size, size, 3), 255, np.uint8)
    patch[y0-top:y1-top, x0-left:x1-left] = image[y0:y1, x0:x1, :3]
    if size == 224:
        return patch
    return np.asarray(Image.fromarray(patch).resize((224, 224), Image.Resampling.BILINEAR))


def prepare_benchmark(root, source, tiffs, arrays, images, panel, arrays_only=False):
    """Never overwrite outputs. Training alone verifies the published gene ranking."""
    root = Path(root)
    genes, mask = read_panel(root/panel)
    loaded = {s: read_source(source, n) for s, n in SLIDES.items()}
    features = loaded['C73_A1'][1]
    for _, other, _, _ in loaded.values():
        if not features.equals(other):
            raise ValueError('Feature tables differ across slides')
    names = features[1].astype(str)
    train = sparse.vstack([loaded[s][0] for s in ['C73_A1', 'C73_B1']], format='csr')
    means = np.asarray(whole_log1p(train).mean(0)).ravel()
    eligible = np.flatnonzero((~names.duplicated() & names.ne('')).to_numpy())
    ranked = eligible[np.argsort(means[eligible])[-200:][::-1]]
    if set(names.iloc[ranked]) != set(genes) or set(names.iloc[ranked[:50]]) != set(genes[mask]):
        raise ValueError('Source training counts do not reproduce the fixed HEG200/50 panel')
    lookup = {name: i for i, name in reversed(list(enumerate(names)))}
    indices = [lookup[g] for g in genes]
    arrays, images = root/arrays, root/images
    paths = [arrays/f'{s}.npz' for s in SLIDES]
    if not arrays_only:
        paths += [images/f'{s}_{size}_patches.npy' for s in SLIDES for size in [224, 896, 1792]]
    if any(path.exists() for path in paths):
        raise FileExistsError('Prepared output exists; use new --arrays/--images directories to audit a rebuild')
    arrays.mkdir(parents=True, exist_ok=True)
    if not arrays_only:
        images.mkdir(parents=True, exist_ok=True)
    manifest = {'genes_sha256': gene_digest(genes), 'training_HEG_ranking_verified': True, 'slides': {}}
    for slide, (counts, _, barcodes, xy) in loaded.items():
        destination = arrays/f'{slide}.npz'
        temp = destination.with_suffix('.tmp.npz')
        np.savez_compressed(temp, raw_counts=counts[:, indices].toarray(),
                            log_normalized=whole_log1p(counts)[:, indices].toarray(),
                            genes=genes, barcodes=barcodes, coordinates_xy=xy)
        temp.replace(destination)
        if not arrays_only:
            # Uncompressed full-resolution TIFFs are memory mapped, not copied into RAM.
            image = tifffile.memmap(Path(tiffs)/f'{slide}.tif')
            if image.ndim != 3 or image.shape[2] < 3 or image.dtype != np.uint8:
                raise ValueError('Expected uint8 RGB full-resolution TIFF')
            for size in [224, 896, 1792]:
                destination = images/f'{slide}_{size}_patches.npy'
                temp = destination.with_suffix('.tmp.npy')
                patches = np.lib.format.open_memmap(temp, mode='w+', dtype=np.uint8, shape=(len(xy),224,224,3))
                for i, (x, y) in enumerate(xy):
                    patches[i] = crop_view(image, x, y, size)
                patches.flush()
                del patches
                temp.replace(destination)
            del image
        manifest['slides'][slide] = {'spots': len(barcodes)}
        print('Prepared', slide, len(barcodes), flush=True)
    (arrays/'preparation.json').write_text(json.dumps(manifest, indent=2), encoding='utf8')
    return manifest
