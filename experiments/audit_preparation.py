"""Compare a fresh prepare --arrays-only rebuild and representative image crops."""
from pathlib import Path
import json
import sys

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from he2st.preprocessing import crop_view


def main():
    rebuilt = Path(sys.argv[1])
    output = {}
    for suffix in ['A1', 'B1', 'C1', 'D1']:
        slide = 'C73_'+suffix
        with np.load(rebuilt/f'{slide}.npz') as new, np.load(ROOT/f'data/processed/gse240429_heg/arrays/{slide}.npz') as old:
            for key in ['raw_counts', 'log_normalized', 'genes', 'barcodes', 'coordinates_xy']:
                np.testing.assert_array_equal(new[key], old[key], err_msg=f'{slide}/{key}')
            xy = new['coordinates_xy']
        image = tifffile.memmap(ROOT/f'data/processed/gse240429/tiff/{slide}.tif')
        ids = np.linspace(0, len(xy)-1, 12, dtype=int)
        for scale in [224, 896, 1792]:
            cache = np.load(ROOT/f'data/processed/gse240429/context_20260913/{slide}_{scale}_patches.npy', mmap_mode='r')
            for i in ids:
                np.testing.assert_array_equal(crop_view(image, *xy[i], scale), cache[i])
        del image
        output[slide] = {'all_array_fields_exact': True, 'exact_patch_checks': len(ids)*3}
    (rebuilt/'verification.json').write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
