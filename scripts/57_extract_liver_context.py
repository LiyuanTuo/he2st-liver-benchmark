"""Image-only context caches. No expression values used; fixed raw-pixel fields of view."""
from pathlib import Path
import json, time
import numpy as np
import tifffile
from PIL import Image
import torch
from torchvision.models import resnet18, ResNet18_Weights

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/processed/gse240429/context_20260913'
OUT.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(6)
torch.backends.cudnn.benchmark=True
torch.manual_seed(42)

def main():
    model=resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc=torch.nn.Identity(); model.cuda().eval()
    mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None]
    std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
    started=time.time()
    for suffix in ['A1','B1','C1','D1']:
        slide='C73_'+suffix
        # Test coordinates/images are allowed at inference; test labels never read here.
        with np.load(ROOT/f'data/processed/gse240429_heg/arrays/{slide}.npz') as data:
            xy=data['coordinates_xy']
        source=tifffile.memmap(ROOT/f'data/processed/gse240429/tiff/{slide}.tif')
        for size in [224,896,1792]:
            dst=OUT/f'{slide}_{size}_resnet18.npy'
            if dst.exists() and np.load(dst,mmap_mode='r').shape==(len(xy),512):
                print('reuse',dst.name,flush=True);continue
            patchfile=OUT/f'{slide}_{size}_patches.npy'
            patches=np.lib.format.open_memmap(patchfile,mode='w+',dtype='uint8',shape=(len(xy),224,224,3))
            if size==224:
                patches[:]=np.load(ROOT/f'data/processed/gse240429/image/{slide}_patches_uint8.npy',mmap_mode='r')
            else:
                for i,(x,y) in enumerate(xy):
                    x,y=int(round(x)),int(round(y)); h=size//2
                    x0,x1=max(0,x-h),min(source.shape[1],x+h)
                    y0,y1=max(0,y-h),min(source.shape[0],y+h)
                    crop=np.full((size,size,3),255,dtype=np.uint8)
                    crop[y0-y+h:y1-y+h,x0-x+h:x1-x+h]=source[y0:y1,x0:x1,:3]
                    patches[i]=np.asarray(Image.fromarray(crop).resize((224,224),Image.Resampling.BILINEAR))
            patches.flush(); rows=[]
            with torch.inference_mode():
                for start in range(0,len(xy),64):
                    batch=torch.from_numpy(patches[start:start+64].copy()).permute(0,3,1,2).cuda().float()/255
                    # Full float32 to keep scale comparisons numerically consistent.
                    rows.append(model((batch-mean)/std).cpu().numpy())
            values=np.concatenate(rows); assert np.isfinite(values).all()
            np.save(dst,values)
            print(slide,size,values.shape,'elapsed',round(time.time()-started,1),flush=True)
        del source
    (OUT/'manifest.json').write_text(json.dumps({'scales_pixels':[224,896,1792],
       'approx_field_um':[57.64,230.56,461.12], 'encoder':'ImageNet ResNet18 frozen',
       'image_resize':'bilinear 224x224', 'labels_used':False},indent=2))

if __name__=='__main__':main()
