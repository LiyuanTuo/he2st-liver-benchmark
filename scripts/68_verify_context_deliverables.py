"""Render and audit the final 11-page PPT/PDF and new numerical claims."""
from pathlib import Path
import json,hashlib
import fitz
from pptx import Presentation
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/liver_context_20260913';FIG=ROOT/'figures/liver_context_20260913'
def main():
    ppt=ROOT/'H&E空间转录组生成研究_统一Benchmark汇报_庹力元.pptx';pdf=ppt.with_suffix('.pdf')
    deck=Presentation(ppt);doc=fitz.open(pdf);assert len(deck.slides)==len(doc)==11
    comparison=json.loads((OUT/'comparison.json').read_text());best=comparison['metrics'][comparison['selected_by_C1']]
    text='\n'.join(p.get_text() for p in doc)
    for v in [f'{best["HEG200"]:.4f}',f'{best["HEG50"]:.4f}','0.1472','0.1978','0.8803','0.8930']:assert v in text,v
    render=json.loads((ROOT/'results/verified_20260913/powerpoint_render_audit.json').read_text(encoding='utf-8-sig'))
    assert not render['TextOverflow'],render['TextOverflow']
    grid=Image.new('RGB',(1860,1480),'white')
    for i,page in enumerate(doc):
        target=FIG/f'slide_{i+1:02d}.png';page.get_pixmap(matrix=fitz.Matrix(1.4,1.4)).save(target)
        thumb=Image.open(target).convert('RGB');thumb.thumbnail((600,338));tile=Image.new('RGB',(620,370),'#dddddd')
        tile.paste(thumb,(10,25));ImageDraw.Draw(tile).text((10,6),str(i+1),fill='black');grid.paste(tile,((i%3)*620,(i//3)*370))
    grid.save(FIG/'all_slides_contact_sheet.png');(OUT/'pdf_text.txt').write_text(text,encoding='utf8')
    paths=[ppt,pdf,ROOT/'docs/06_LIVER_CONTEXT_IMPROVEMENT.md',OUT/'comparison.json',OUT/'verification.json',OUT/'fresh_checkpoint_verification.json']
    audit=dict(slides=11,models=len(comparison['metrics']),PPT_PDF_numbers_verified=True,powerpoint_text_overflow=False,
      files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    (OUT/'final_deliverable_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf8');print('PASS',audit)
if __name__=='__main__':main()
