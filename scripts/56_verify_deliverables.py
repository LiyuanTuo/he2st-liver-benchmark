"""Check final PPT/PDF numbers and prediction shapes; render local review images."""
from pathlib import Path
import hashlib
import json
import fitz
import numpy as np
from PIL import Image, ImageDraw
from pptx import Presentation

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/verified_20260913'
FIG = ROOT / 'figures/verified_20260913'
ppt = ROOT / 'H&E空间转录组生成研究_统一Benchmark汇报_庹力元.pptx'
pdf = ppt.with_suffix('.pdf')
document = fitz.open(pdf)
deck = Presentation(ppt)
assert len(document) == len(deck.slides) == 11
benchmark = json.loads((OUT / 'benchmark.json').read_text(encoding='utf8'))
assert not benchmark['pending'] and len(benchmark['methods']) == 7
with np.load(OUT / 'benchmark_predictions.npz') as predictions:
    assert len(predictions['genes']) == 200 and len(predictions['barcodes']) == 2265
    for method in benchmark['methods']:
        assert predictions[method].shape == (2265, 200)
        assert np.isfinite(predictions[method]).all()
text = '\n'.join(page.get_text() for page in document)
for value in ['0.1472', '0.1978', '0.8803', '0.8930']:
    assert value in text
assert '训练中预留' not in text and '随机 基因' not in text
render = json.loads((OUT / 'powerpoint_render_audit.json').read_text(encoding='utf-8-sig'))
assert not render['TextOverflow']
grid = Image.new('RGB', (1860, 1480), 'white')
for i, page in enumerate(document):
    target = FIG / f'slide_{i+1:02d}.png'
    page.get_pixmap(matrix=fitz.Matrix(1.4, 1.4)).save(target)
    thumb = Image.open(target).convert('RGB')
    thumb.thumbnail((600, 338))
    tile = Image.new('RGB', (620, 370), '#dddddd')
    tile.paste(thumb, (10, 25))
    ImageDraw.Draw(tile).text((10, 6), str(i+1), fill='black')
    grid.paste(tile, ((i % 3)*620, (i // 3)*370))
grid.save(FIG / 'all_slides_contact_sheet.png')
(OUT / 'pdf_text.txt').write_text(text, encoding='utf8')
files = [ppt, pdf, ROOT / 'docs/05_PCC_DIAGNOSIS.md', OUT / 'benchmark.json']
audit = dict(slides=11, methods=7, prediction_shape=[2265, 200],
    PPT_PDF_numbers_verified=True, all_prediction_values_finite=True,
    powerpoint_text_overflow=False,
    files={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
(OUT / 'final_deliverable_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf8')
print('PASS: 11 slides, 7 methods, 2265 x 200 predictions, PDF numbers, no text overflow.')
