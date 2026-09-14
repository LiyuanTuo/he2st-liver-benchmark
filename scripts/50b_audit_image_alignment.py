"""Verify TIFF extent, spot crops and actual float-to-PIL corruption."""
from pathlib import Path
import json
import pickle
import numpy as np
import tifffile
from torchvision.transforms import ToPILImage

ROOT = Path(__file__).resolve().parents[1]
out = {}
for slide in ['C73_A1', 'C73_B1', 'C73_C1', 'C73_D1']:
    path = ROOT / f'data/processed/gse240429/tiff/{slide}.tif'
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        end = max(o + n for o, n in zip(page.dataoffsets, page.databytecounts))
        assert end <= path.stat().st_size
        shape = page.shape
    image = tifffile.memmap(path)
    patches = np.load(ROOT / f'data/processed/gse240429/image/{slide}_patches_uint8.npy', mmap_mode='r')
    data = np.load(ROOT / f'data/processed/gse240429_heg/arrays/{slide}.npz')
    assert patches.dtype == np.uint8
    for index in np.linspace(0, len(patches) - 1, 32, dtype=int):
        x, y = np.rint(data['coordinates_xy'][index]).astype(int)
        np.testing.assert_array_equal(patches[index], image[y-112:y+112, x-112:x+112, :3])
    out[slide] = dict(bytes=path.stat().st_size, shape=shape,
                     all_strip_bytes_within_file=True, sampled_patches_exactly_match_source=32)
    print(slide, 'PASS', flush=True)
    del image
with open(ROOT / 'data/official_ressat_example/Section_1/dataset.pkl', 'rb') as f:
    section = pickle.load(f)
patch = section[0][0]
original = patch.astype(np.uint8)
incorrect = np.array(ToPILImage()(patch))
correct = np.array(ToPILImage()(original))
np.testing.assert_array_equal(correct, original)
out['official_float_patch'] = dict(dtype=str(patch.dtype), minimum=float(patch.min()), maximum=float(patch.max()),
    changed_pixel_channel_fraction=float((incorrect != original).mean()),
    mean_absolute_channel_error=float(np.abs(incorrect.astype(float) - original.astype(float)).mean()),
    uint8_conversion_matches_source=True,
    scope='Historical official-example input incompatibility; current liver uint8 cache is not affected')
(ROOT / 'results/verified_20260913/tiff_patch_audit.json').write_text(json.dumps(out, indent=2), encoding='utf8')
print(out['official_float_patch'], flush=True)
