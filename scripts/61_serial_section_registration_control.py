"""Explicit same-donor serial-section reference control, NOT a de novo CNN.

Register H&E only, transfer TRAIN RNA; validation/test RNA never used in image
registration. This reference-dependent method cannot establish unseen-patient
generalization and is reported separately from image-to-expression networks.
"""
from pathlib import Path
import gzip,json,time
import cv2
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/liver_registration_20260913';OUT.mkdir(parents=True,exist_ok=True)
cv2.setNumThreads(4)
AUDIT=pd.read_csv(ROOT/'results/verified_20260913/fixed_HEG200_gene_audit.csv');MASK=AUDIT.is_primary_HEG50.to_numpy(bool)
def norm(x):return np.log1p(1e4*x/np.maximum(x.sum(1,keepdims=True),1e-12))
def score(y,p):
    a=y-y.mean(0);b=p-p.mean(0);r=(a*b).sum(0)/np.maximum(np.sqrt((a*a).sum(0)*(b*b).sum(0)),1e-12)
    return dict(HEG200=float(r.mean()),HEG50=float(r[MASK].mean()),MAE=float(np.abs(y-p).mean()))
def image_data(s):
    raw=ROOT/'data/raw/GSE240429'
    image=cv2.imdecode(np.frombuffer(gzip.open(next(raw.glob(f'*C73{s}_tissue_hires_image.png.gz')),'rb').read(),np.uint8),cv2.IMREAD_COLOR)
    scalef=json.loads(gzip.open(next(raw.glob(f'*C73{s}_scalefactors_json.json.gz')),'rt').read())
    with np.load(ROOT/f'data/processed/gse240429_heg/arrays/C73_{s}.npz') as d:xy=d['coordinates_xy']*scalef['tissue_hires_scalef']
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    mask=(gray<180).astype('uint8');n,lab,stats,cent=cv2.connectedComponentsWithStats(mask)
    mask=(lab==(1+stats[1:,cv2.CC_STAT_AREA].argmax())).astype('uint8')*255
    mask=cv2.dilate(mask,np.ones((11,11),np.uint8))
    detector=cv2.SIFT_create(nfeatures=10000,contrastThreshold=.02)
    points,des=detector.detectAndCompute(gray,mask)
    return dict(image=image,xy=xy,points=points,des=des,scale=scalef,mask=mask)
def main():
    train={s:image_data(s) for s in ['A1','B1']}
    ytrain={s:norm(np.load(ROOT/f'data/processed/gse240429_heg/arrays/C73_{s}.npz')['raw_counts']) for s in train}
    diagnostics={};predictions={};locked=None
    for query in ['C1','D1']:
        q=image_data(query);neighbors={};diagnostics[query]={}
        for reference,t in train.items():
            matches=cv2.BFMatcher().knnMatch(q['des'],t['des'],k=2)
            good=[m for m,n in matches if m.distance<.75*n.distance]
            src=np.float32([q['points'][m.queryIdx].pt for m in good]);dst=np.float32([t['points'][m.trainIdx].pt for m in good])
            affine,inliers=cv2.estimateAffine2D(src,dst,method=cv2.RANSAC,ransacReprojThreshold=6,maxIters=10000,confidence=.999)
            sift_inliers=0 if inliers is None else int(inliers.sum());ecc=None
            if affine is None or sift_inliers<15:
                # Adjacent sections do not preserve individual cell keypoints.
                # Register coarse vessels/tissue boundaries via image-only ECC.
                def registration_image(item):
                    m=cv2.resize(item['mask'],(512,512),interpolation=cv2.INTER_NEAREST)>0
                    g=cv2.resize(cv2.cvtColor(item['image'],cv2.COLOR_BGR2GRAY),(512,512)).astype('float32')/255
                    g=np.where(m,1-g,0).astype('float32')
                    g=cv2.GaussianBlur(g,(0,0),2)
                    yy,xx=np.where(m);return g,np.array([xx.mean(),yy.mean()]),float(m.sum())
                ti,tc,ta=registration_image(t);qi,qc,qa=registration_image(q)
                scale=np.sqrt(qa/ta);warp=np.array([[scale,0,qc[0]-scale*tc[0]],[0,scale,qc[1]-scale*tc[1]]],dtype='float32')
                ecc,warp=cv2.findTransformECC(ti,qi,warp,cv2.MOTION_AFFINE,(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,600,1e-6),None,5)
                if ecc<.5:raise RuntimeError(f'Poor image-only ECC registration {query}->{reference}: {ecc}')
                w=np.vstack([warp,[0,0,1]])
                qscale=np.diag([512/q['image'].shape[1],512/q['image'].shape[0],1])
                tscale=np.diag([t['image'].shape[1]/512,t['image'].shape[0]/512,1])
                affine=(tscale@np.linalg.inv(w)@qscale)[:2]
            mapped=q['xy']@affine[:,:2].T+affine[:,2]
            dist,idx=NearestNeighbors(n_neighbors=12).fit(t['xy']).kneighbors(mapped)
            neighbors[reference]=(dist,idx)
            residual=np.linalg.norm(src@affine[:,:2].T+affine[:,2]-dst,axis=1)
            diagnostics[query][reference]=dict(matches=len(good),sift_inliers=sift_inliers,ecc=ecc,affine=affine.tolist(),determinant=float(np.linalg.det(affine[:,:2])),median_nearest_spot_distance=float(np.median(dist[:,0])))
            rendered=cv2.drawMatches(q['image'],q['points'],t['image'],t['points'],[m for m,flag in zip(good,inliers.ravel()) if flag][:60],None,flags=2)
            cv2.imwrite(str(OUT/f'{query}_to_{reference}_matches.jpg'),rendered)
            registered=cv2.warpAffine(q['image'],affine,(t['image'].shape[1],t['image'].shape[0]),borderValue=(255,255,255))
            cv2.imwrite(str(OUT/f'{query}_to_{reference}_overlay.jpg'),cv2.addWeighted(t['image'],.5,registered,.5,0))
        candidates={}
        for refs in [('A1',),('B1',),('A1','B1')]:
            for k in [1,3,6,12]:
                estimates=[]
                for ref in refs:
                    distance,ids=neighbors[ref];weight=1/np.maximum(distance[:,:k],1)**2;weight/=weight.sum(1,keepdims=True)
                    estimates.append((ytrain[ref][ids[:,:k]]*weight[:,:,None]).sum(1))
                candidates[f'{"+".join(refs)}_k{k}']=np.mean(estimates,axis=0)
        if query=='C1':
            true=norm(np.load(ROOT/'data/processed/gse240429_heg/arrays/C73_C1.npz')['raw_counts'])
            validation={n:score(true,p) for n,p in candidates.items()};selected=max(validation,key=lambda n:validation[n]['HEG200'])
            locked=dict(selected=selected,validation=validation,locked_unix=time.time(),limitation='Requires spatially corresponding serial-section TRAIN H&E+RNA reference; not a general de novo image model.')
            (OUT/'selection_locked.json').write_text(json.dumps(locked,indent=2))
        else:
            d=dict(np.load(ROOT/'data/processed/gse240429_heg/arrays/C73_D1.npz'));true=norm(d['raw_counts'])
            metrics=score(true,candidates[selected]);print('TEST',selected,metrics,flush=True)
            (OUT/'test_metrics.json').write_text(json.dumps(dict(selected=selected,metrics=metrics),indent=2))
            np.savez_compressed(OUT/'D1_prediction.npz',truth=true,predicted=candidates[selected],genes=d['genes'],coordinates_xy=d['coordinates_xy'],barcodes=d['barcodes'])
        print(query,diagnostics[query],flush=True)
    (OUT/'registration_diagnostics.json').write_text(json.dumps(diagnostics,indent=2))

if __name__=='__main__':main()
