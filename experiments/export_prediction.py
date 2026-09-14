"""Export an RNA-free prediction NPZ as AnnData with optional H&E overlay."""
from pathlib import Path
import argparse
import gzip
import json

import anndata as ad
import numpy as np
import pandas as pd
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]


def open_file(path, mode):
    return gzip.open(path,mode) if path.suffix=='.gz' else path.open(mode)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prediction',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--slide',default='C73_D1')
    ap.add_argument('--hires',type=Path)
    ap.add_argument('--scalefactors',type=Path)
    args=ap.parse_args()
    if bool(args.hires)!=bool(args.scalefactors):
        ap.error('H&E hires and scalefactors must be supplied together')
    with np.load(args.prediction) as d:
        var=pd.read_csv(ROOT/'benchmarks/liver/genes.csv').set_index('gene')
        np.testing.assert_array_equal(var.index.to_numpy(str),d['genes'])
        obs=pd.DataFrame(index=pd.Index(d['barcodes'].astype(str),name='barcode'))
        obs['slide']=args.slide
        value=ad.AnnData(X=d['predicted'].astype(np.float32),obs=obs,var=var)
        value.obsm['spatial']=d['coordinates_xy']
        value.uns['expression_units']='Predicted log1p(10000 * HEG200-relative abundance); not raw UMI'
        value.uns['RNA_truth_included']=False
        if args.hires:
            with open_file(args.hires,'rb') as f:
                image=np.asarray(Image.open(f).convert('RGB'))
            with open_file(args.scalefactors,'rt') as f:
                scales=json.load(f)
            value.uns['spatial']={args.slide:{'images':{'hires':image},'scalefactors':scales}}
        args.output.parent.mkdir(parents=True,exist_ok=True)
        value.write_h5ad(args.output,compression='gzip')
        reopened=ad.read_h5ad(args.output)
        np.testing.assert_array_equal(reopened.X,d['predicted'])
        assert not reopened.layers and reopened.raw is None
        print(args.output,reopened.shape,'prediction only')


if __name__=='__main__':main()
