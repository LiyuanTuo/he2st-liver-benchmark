"""Fixed HEG200/50; train A1+B1, choose only on C1, load D1 labels after lock.

ContextResNet: frozen ImageNet ResNet18 at 58/231/461 um, image-only
neighbor pooling, and a newly trained residual nonlinear regression head.
All main scores use the original unprojected panel-relative target.
"""
from pathlib import Path
import copy, hashlib, json, time
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/liver_context_20260913'; OUT.mkdir(parents=True,exist_ok=True)
CACHE=ROOT/'data/processed/gse240429/context_20260913'
torch.set_num_threads(6); torch.backends.cuda.matmul.allow_tf32=False
DEVICE='cuda'

def norm(x):
    x=np.maximum(np.asarray(x,dtype=np.float64),0)
    return np.log1p(10000*x/np.maximum(x.sum(1,keepdims=True),1e-12)).astype('float32')

def pcc(y,p):
    y=y.astype('float64')-y.mean(0);p=p.astype('float64')-p.mean(0)
    d=np.sqrt((y*y).sum(0)*(p*p).sum(0))
    return np.divide((y*p).sum(0),d,out=np.full(y.shape[1],np.nan),where=d>1e-12)

AUDIT=pd.read_csv(ROOT/'results/verified_20260913/fixed_HEG200_gene_audit.csv')
MASK=AUDIT.is_primary_HEG50.to_numpy(bool)
def score(y,p):
    r=pcc(y,p)
    return dict(HEG200=float(np.nanmean(r)),HEG50=float(np.nanmean(r[MASK])),
                MAE=float(np.abs(y-p).mean()),effective=int(np.isfinite(r).sum()))

def data(s):
    d=dict(np.load(ROOT/f'data/processed/gse240429_heg/arrays/C73_{s}.npz'))
    np.testing.assert_array_equal(d['genes'],AUDIT.gene)
    return d

def image_features(s):
    xs=[np.load(CACHE/f'C73_{s}_{size}_resnet18.npy') for size in [224,896,1792]]
    # Geometry only: no expression-derived graph, no cross-section registration.
    with np.load(ROOT/f'data/processed/gse240429_heg/arrays/C73_{s}.npz') as d:xy=d['coordinates_xy']
    ids=NearestNeighbors(n_neighbors=17).fit(xy).kneighbors(xy,return_distance=False)[:,1:]
    multi=np.concatenate(xs,1)
    return {'single':xs[0],'multi':multi,'graph':np.concatenate([multi,multi[ids].mean(1)],1)},ids

def tensor(x):return torch.as_tensor(x,dtype=torch.float32,device=DEVICE)
def savejson(name,obj): (OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False))

class ContextHead(nn.Module):
    def __init__(self,n):
        super().__init__()
        self.linear=nn.Linear(n,200)
        self.nonlinear=nn.Sequential(nn.Linear(n,256),nn.LayerNorm(256),nn.GELU(),nn.Dropout(.3),
            nn.Linear(256,128),nn.GELU(),nn.Dropout(.2),nn.Linear(128,200))
    def forward(self,x):return self.linear(x)+self.nonlinear(x)

def main():
    started=time.time()
    savejson('protocol.json',dict(train=['A1','B1'],validation='C1',test='D1',
      target='unprojected log1p(1e4*count/sum HEG200 counts)',selection='C1 macro HEG200 PCC only',
      genes_sha256=hashlib.sha256(('\n'.join(AUDIT.gene)+'\n').encode()).hexdigest(),
      ridge_alphas=[100,1000,10000,100000],smoothing=[0,.5,.8],
      targets=['whole','panel','panel_train_smooth_50percent_neighbor6'],
      rbf_centers=1024,rbf_bandwidth=[.5,1,2],rbf_alpha=[.1,1,10,100],
      neural_seeds=[42,17,83],neural_max_epochs=200,neural_patience=30,
      note='D1 has been evaluated historically; this is a development test, not a new independent cohort.'))
    ds=[data(s) for s in ['A1','B1','C1']]
    yt=norm(np.concatenate([d['raw_counts'] for d in ds[:2]])); yv=norm(ds[2]['raw_counts'])
    native=np.concatenate([d['log_normalized'] for d in ds[:2]])
    fs=[image_features(s) for s in ['A1','B1','C1']]
    results={}; fitted={}; val_predictions={}; feature_info={}
    # Matched target and context ablations; retain every result, no test filtering.
    for feature in ['single','multi','graph']:
        x=np.concatenate([f[0][feature] for f in fs[:2]])
        v=fs[2][0][feature]; xm=x.mean(0); xs=np.maximum(x.std(0),.1)
        x=tensor((x-xm)/xs); v=tensor((v-xm)/xs)
        feature_info[feature]=(xm,xs)
        gram=x.T@x; eye=torch.eye(x.shape[1],device=DEVICE)
        smooth_labels=np.concatenate([common_y[f[1][:,:6]].mean(1) for common_y,f in zip([yt[:len(ds[0]['raw_counts'])],yt[len(ds[0]['raw_counts']):]],fs[:2])])
        for target,labels in [('whole',native),('panel',yt),('panel_train_smooth',.5*yt+.5*smooth_labels)]:
            ym=labels.mean(0); cross=x.T@tensor(labels-ym); best=None
            for alpha in [100,1000,10000,100000]:
                coef=torch.linalg.solve(gram+alpha*eye,cross)
                val=(v@coef).cpu().numpy()+ym
                if target=='whole':val=norm(np.expm1(val))
                for smooth in [0,.5,.8]:
                    pred=(1-smooth)*val+smooth*val[fs[2][1]].mean(1)
                    result=score(yv,pred)
                    if best is None or result['HEG200']>best['validation']['HEG200']:
                        best=dict(alpha=alpha,smooth=smooth,validation=result)
                        bestcoef=coef.cpu().numpy(); bestval=pred.copy()
            name=f'ridge_{feature}_{target}'
            fitted[name]=dict(feature=feature,target=target,coef=bestcoef,ym=ym,**best)
            results[name]=best;val_predictions[name]=bestval
            print(name,best,flush=True);savejson('validation.json',results)
        del gram,eye,x,v
    feature=max(['single','multi','graph'],key=lambda f:results[f'ridge_{f}_panel']['validation']['HEG200'])
    xm,xs=feature_info[feature]
    xt=tensor((np.concatenate([f[0][feature] for f in fs[:2]])-xm)/xs)
    xv=tensor((fs[2][0][feature]-xm)/xs)
    # Nonlinear image-only control: 1024 radial basis centers sampled from train.
    torch.manual_seed(42)
    anchors=xt[torch.randperm(len(xt),device=DEVICE)[:1024]]
    dt=torch.cdist(xt,anchors).square()/xt.shape[1]
    dv=torch.cdist(xv,anchors).square()/xt.shape[1]
    median=float(dt.median());rbf_best=None
    for bandwidth in [.5,1,2]:
        z=torch.exp(-dt/(median*bandwidth));zv=torch.exp(-dv/(median*bandwidth))
        zm=z.mean(0);z=z-zm;zv=zv-zm
        gram=z.T@z;cross=z.T@tensor(yt-yt.mean(0))
        for alpha in [.1,1,10,100]:
            coef=torch.linalg.solve(gram+alpha*torch.eye(1024,device=DEVICE),cross)
            pred=(zv@coef).cpu().numpy()+yt.mean(0)
            for a in [0,.5,.8]:
                pv=(1-a)*pred+a*pred[fs[2][1]].mean(1);metric=score(yv,pv)
                if rbf_best is None or metric['HEG200']>rbf_best['validation']['HEG200']:
                    rbf_best=dict(bandwidth=bandwidth,alpha=alpha,smooth=a,validation=metric)
                    rbf_state=dict(anchors=anchors.cpu().numpy(),median=median,zm=zm.cpu().numpy(),coef=coef.cpu().numpy(),ym=yt.mean(0))
                    rbf_val=pv.copy()
    results['Context_RBF']=rbf_best;val_predictions['Context_RBF']=rbf_val
    np.savez_compressed(OUT/'Context_RBF_weights.npz',xm=xm,xs=xs,**rbf_state)
    print('Context_RBF',rbf_best,flush=True)
    del dt,dv,z,zv,gram,cross,coef
    ym=yt.mean(0);ys=np.maximum(yt.std(0),.1);tz=tensor((yt-ym)/ys)
    neural_models=[]; neural_vals=[]
    for seed in [42,17,83]:
        torch.manual_seed(seed);np.random.seed(seed)
        model=ContextHead(xt.shape[1]).cuda()
        opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.02)
        best=-1;stale=0;history=[]
        for epoch in range(1,201):
            model.train();order=torch.randperm(len(xt),device=DEVICE);losses=[]
            for batch in order.split(256):
                z=model(xt[batch]);t=tz[batch]
                zc=z-z.mean(0);tc=t-t.mean(0)
                corr=(zc*tc).sum(0)/(zc.square().sum(0)*tc.square().sum(0)+1e-6).sqrt()
                loss=(z-t).square().mean()+.2*(1-corr.mean())
                opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
                losses.append(loss.item())
            model.eval()
            with torch.inference_mode():pv=model(xv).cpu().numpy()*ys+ym
            metric=score(yv,pv);history.append(dict(epoch=epoch,loss=float(np.mean(losses)),**metric))
            if metric['HEG200']>best+1e-5:
                best=metric['HEG200'];stale=0;state=copy.deepcopy(model.state_dict());bestepoch=epoch
            else:stale+=1
            if epoch%10==0:print('neural',seed,epoch,metric,'elapsed',round(time.time()-started),flush=True)
            if stale>=30:break
        model.load_state_dict(state);model.eval()
        with torch.inference_mode():pv=model(xv).cpu().numpy()*ys+ym;pt=model(xt).cpu().numpy()*ys+ym
        name=f'ContextResNet_seed{seed}'
        results[name]=dict(feature=feature,epoch=bestepoch,validation=score(yv,pv),train=score(yt,pt))
        neural_models.append(model);neural_vals.append(pv);val_predictions[name]=pv
        torch.save(dict(model=state,feature=feature,xm=xm,xs=xs,ym=ym,ys=ys,epoch=bestepoch),OUT/f'{name}.pt')
        savejson(f'{name}_training.json',history);savejson('validation.json',results)
    ensemble=np.mean(neural_vals,0)
    smooth=max([0,.5,.8],key=lambda a:score(yv,(1-a)*ensemble+a*ensemble[fs[2][1]].mean(1))['HEG200'])
    neural=(1-smooth)*ensemble+smooth*ensemble[fs[2][1]].mean(1)
    results['ContextResNet_ensemble']=dict(smooth=smooth,feature=feature,validation=score(yv,neural))
    val_predictions['ContextResNet_ensemble']=neural
    ridge_name=max(fitted,key=lambda n:results[n]['validation']['HEG200'])
    mix=max([0,.25,.5,.75,1],key=lambda a:score(yv,a*neural+(1-a)*val_predictions[ridge_name])['HEG200'])
    results['ContextResNet_hybrid']=dict(neural_weight=mix,ridge=ridge_name,validation=score(yv,mix*neural+(1-mix)*val_predictions[ridge_name]))
    val_predictions['ContextResNet_hybrid']=mix*neural+(1-mix)*val_predictions[ridge_name]
    selected=max(results,key=lambda n:results[n]['validation']['HEG200'])
    # Persist the choice before accessing test expression, all tested models retained.
    savejson('selection_locked.json',dict(selected=selected,validation=results,locked_unix=time.time()))
    np.savez_compressed(OUT/'C1_predictions.npz',truth=yv,genes=AUDIT.gene.to_numpy(str),**val_predictions)
    ft,neighbors=image_features('D1');testdata=data('D1');yd=norm(testdata['raw_counts']);preds={}
    for name,fit in fitted.items():
        mean,std=feature_info[fit['feature']]
        pred=((tensor((ft[fit['feature']]-mean)/std)@tensor(fit['coef'])).cpu().numpy()+fit['ym'])
        if fit['target']=='whole':pred=norm(np.expm1(pred))
        a=fit['smooth'];preds[name]=(1-a)*pred+a*pred[neighbors].mean(1)
        np.savez_compressed(OUT/f'{name}_weights.npz',xm=mean,xs=std,coef=fit['coef'],ym=fit['ym'])
    xx=tensor((ft[feature]-xm)/xs)
    z=torch.exp(-torch.cdist(xx,tensor(rbf_state['anchors'])).square()/xx.shape[1]/(median*rbf_best['bandwidth']))-tensor(rbf_state['zm'])
    p=(z@tensor(rbf_state['coef'])).cpu().numpy()+rbf_state['ym'];a=rbf_best['smooth']
    preds['Context_RBF']=(1-a)*p+a*p[neighbors].mean(1)
    for seed,model in zip([42,17,83],neural_models):
        with torch.inference_mode():preds[f'ContextResNet_seed{seed}']=model(xx).cpu().numpy()*ys+ym
    p=np.mean([preds[f'ContextResNet_seed{s}'] for s in [42,17,83]],0)
    preds['ContextResNet_ensemble']=(1-smooth)*p+smooth*p[neighbors].mean(1)
    preds['ContextResNet_hybrid']=mix*preds['ContextResNet_ensemble']+(1-mix)*preds[ridge_name]
    test={name:score(yd,p) for name,p in preds.items()}
    savejson('test_metrics.json',dict(selected=selected,models=test,elapsed_seconds=time.time()-started))
    np.savez_compressed(OUT/'D1_predictions.npz',truth=yd,genes=testdata['genes'],barcodes=testdata['barcodes'],coordinates_xy=testdata['coordinates_xy'],**preds)
    pd.DataFrame({'gene':AUDIT.gene,'HEG50':MASK,**{name:pcc(yd,p) for name,p in preds.items()}}).to_csv(OUT/'D1_per_gene.csv',index=False)
    print('LOCKED SELECTION',selected,'TEST',test,flush=True)

if __name__=='__main__':main()
