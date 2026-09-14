"""Prepared image/spot data and the fixed, training-selected gene panel."""
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def panel_log1p(counts):
    counts = np.asarray(counts, dtype=np.float64)
    if counts.ndim != 2 or not np.isfinite(counts).all() or (counts < 0).any():
        raise ValueError('Counts must be a finite nonnegative spot-by-gene matrix')
    return np.log1p(10000 * counts / np.maximum(counts.sum(1, keepdims=True), 1e-12)).astype(np.float32)


def gene_digest(genes):
    return hashlib.sha256(('\n'.join(genes) + '\n').encode()).hexdigest()


def read_panel(path):
    frame = pd.read_csv(path)
    genes = frame['gene'].to_numpy(str)
    mask = frame['is_primary_HEG50'].to_numpy(bool)
    if len(genes) != 200 or len(set(genes)) != len(genes) or mask.sum() != 50:
        raise ValueError('Expected 200 unique fixed genes and exactly 50 HEGs')
    return genes, mask


class SlideDataset(Dataset):
    """Prediction mode reads only images and row/column metadata, never RNA."""

    def __init__(self, root, slides, arrays, images, scales, genes, labels=True):
        self.slides = list(slides)
        if not self.slides or len(set(slides)) != len(slides):
            raise ValueError('Slides must be nonempty and unique')
        self.patches, self.metadata, self.targets, self.xy = [], [], [], []
        self.sizes = []
        for slide in slides:
            with np.load(Path(root) / arrays / f'{slide}.npz') as d:
                np.testing.assert_array_equal(d['genes'], genes)
                meta = {key: d[key] for key in ['genes', 'barcodes', 'coordinates_xy']}
                if len(set(meta['barcodes'])) != len(meta['barcodes']):
                    raise ValueError(f'Duplicate barcodes in {slide}')
                if labels:
                    self.targets.append(panel_log1p(d['raw_counts']))
            patches = [np.load(Path(root) / images / f'{slide}_{scale}_patches.npy', mmap_mode='r') for scale in scales]
            n = len(meta['barcodes'])
            for p in patches:
                if p.shape != (n, 224, 224, 3) or p.dtype != np.uint8:
                    raise ValueError(f'Invalid uint8 patch cache for {slide}: {p.shape}')
            self.patches.append(patches)
            self.metadata.append(meta)
            self.xy.append(meta['coordinates_xy'])
            self.sizes.append(n)
        self.offsets = np.cumsum(self.sizes)
        self.y = np.concatenate(self.targets) if labels else None

    def __len__(self):
        return int(self.offsets[-1])

    def __getitem__(self, index):
        slide = int(np.searchsorted(self.offsets, index, side='right'))
        local = index - (int(self.offsets[slide-1]) if slide else 0)
        images = np.stack([p[local] for p in self.patches[slide]])
        image = torch.from_numpy(images).permute(0, 3, 1, 2)
        if self.y is None:
            return image
        return image, torch.from_numpy(self.y[index])


def training_neighbor_targets(dataset, fraction):
    """Optional label denoising inside each training slide; no test/val graph."""
    from sklearn.neighbors import NearestNeighbors
    if not 0 <= fraction <= 1 or dataset.y is None:
        raise ValueError('Invalid training target smoothing')
    if fraction == 0:
        return dataset.y.copy()
    values = []
    for y, xy in zip(dataset.targets, dataset.xy):
        neighbors = min(6, len(y)-1)
        if neighbors < 1:
            raise ValueError('Need at least two training spots for smoothing')
        ids = NearestNeighbors(n_neighbors=neighbors+1).fit(xy).kneighbors(xy, return_distance=False)
        ids = np.array([row[row != i][:neighbors] for i, row in enumerate(ids)])
        values.append((1-fraction)*y + fraction*y[ids].mean(1))
    return np.concatenate(values).astype(np.float32)
