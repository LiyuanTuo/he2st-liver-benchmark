"""Export the selected H&E prediction as usable AnnData, without test RNA."""
from pathlib import Path
import gzip,json
import anndata as ad
import numpy as np
import pandas as pd
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/liver_context_20260913'
def main():
    result=json.loads((OUT/'comparison.json').read_text());selected=result['selected_by_C1']
    d=np.load(OUT/'combined_predictions.npz');audit=pd.read_csv(ROOT/'results/verified_20260913/fixed_HEG200_gene_audit.csv')
    np.testing.assert_array_equal(audit.gene,d['genes'])
    var=audit.set_index('gene');var.index=var.index.astype(str)
    obs=pd.DataFrame(index=pd.Index(d['barcodes'].astype(str),name='barcode'));obs['slide']='C73_D1'
    a=ad.AnnData(X=d[selected].astype('float32'),obs=obs,var=var)
    a.obsm['spatial']=d['coordinates_xy'].astype('float64')
    a.uns['prediction_model']=selected
    a.uns['expression_units']='Predicted log1p(10000 * HEG200-relative abundance); not UMI counts'
    a.uns['training_slides']='C73_A1+C73_B1';a.uns['validation_slide']='C73_C1'
    a.uns['gene_selection']='Top200 mean whole-transcriptome log expression on training only; HEG50 fixed first50'
    raw=ROOT/'data/raw/GSE240429'
    image=np.asarray(Image.open(gzip.open(next(raw.glob('*C73D1_tissue_hires_image.png.gz')))))
    scalef=json.loads(gzip.open(next(raw.glob('*C73D1_scalefactors_json.json.gz')),'rt').read())
    a.uns['spatial']={'C73_D1':{'images':{'hires':image},'scalefactors':scalef}}
    destination=OUT/'C73_D1_HE_predicted_HEG200.h5ad';a.write_h5ad(destination,compression='gzip')
    reopened=ad.read_h5ad(destination);assert reopened.shape==(2265,200)
    np.testing.assert_array_equal(reopened.X,d[selected]);np.testing.assert_array_equal(reopened.var_names,d['genes'])
    assert not reopened.layers and 'truth' not in reopened.uns
    print(destination,reopened.shape,'prediction only, with H&E overlay metadata')
if __name__=='__main__':main()
