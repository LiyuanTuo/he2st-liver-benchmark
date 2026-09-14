"""New shared multiscale CNN, end-to-end, fixed panel-relative targets.

Two preregistered arms: random initialization (entire model from scratch) and
ImageNet initialization. Both retrain all parameters using only A1+B1.
Validation C1 selects epoch/TTA; D1 is opened only after BOTH arms are locked.
"""
from pathlib import Path
import argparse, copy, json, random, time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,Dataset
from torchvision.models import resnet18,ResNet18_Weights
import importlib.util

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('context',ROOT/'scripts/58_liver_context_experiments.py')
common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common)
OUT=ROOT/'results/liver_context_fusion_20260913';OUT.mkdir(parents=True,exist_ok=True)
ARMS=['imagenet','scratch']
SEED=42
torch.set_num_threads(6);torch.backends.cudnn.benchmark=True

class Slides(Dataset):
    def __init__(self,slides):
        self.arrays=[common.data(s) for s in slides]
        self.y=common.norm(np.concatenate([d['raw_counts'] for d in self.arrays]))
        self.patches=[[np.load(common.CACHE/f'C73_{s}_{scale}_patches.npy',mmap_mode='r') for scale in [224,896,1792]] for s in slides]
        self.sizes=np.cumsum([len(p[0]) for p in self.patches])
    def __len__(self):return len(self.y)
    def __getitem__(self,i):
        s=int(np.searchsorted(self.sizes,i,side='right'));j=i-(self.sizes[s-1] if s else 0)
        x=np.stack([p[j] for p in self.patches[s]])
        return torch.from_numpy(x).permute(0,3,1,2),torch.from_numpy(self.y[i])

class ContextFusion(nn.Module):
    def __init__(self,pretrained):
        super().__init__()
        self.encoder=resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.encoder.fc=nn.Identity()
        self.head=nn.Sequential(nn.LayerNorm(1536),nn.Dropout(.3),nn.Linear(1536,256),nn.GELU(),nn.Dropout(.2),nn.Linear(256,200))
    def forward(self,x):
        b,s,c,h,w=x.shape
        f=self.encoder(x.reshape(b*s,c,h,w))
        return self.head(f.reshape(b,s*512))

def main():
    torch.manual_seed(SEED);np.random.seed(SEED);random.seed(SEED)
    train=Slides(['A1','B1']);val=Slides(['C1'])
    tl=DataLoader(train,batch_size=24,shuffle=True,num_workers=2,pin_memory=True,persistent_workers=True)
    vl=DataLoader(val,batch_size=32,num_workers=2,pin_memory=True,persistent_workers=True)
    ym=torch.tensor(train.y.mean(0),device='cuda');ys=torch.tensor(np.maximum(train.y.std(0),.1),device='cuda')
    im=torch.tensor([.485,.456,.406],device='cuda')[None,None,:,None,None]
    sd=torch.tensor([.229,.224,.225],device='cuda')[None,None,:,None,None]
    def prepare(x,augment=False):
        x=x.cuda(non_blocking=True).float()/255
        if augment:
            x=torch.rot90(x,random.randrange(4),[-2,-1])
            if random.random()<.5:x=x.flip(-1)
            # Mild brightness/contrast preserves cell structure; same transform across scales.
            x=((x-.5)*(1+random.uniform(-.1,.1))+.5+random.uniform(-.03,.03)).clamp(0,1)
        return (x-im)/sd
    @torch.inference_mode()
    def predict(model,loader,tta=False):
        model.eval();out=[]
        for x,y in loader:
            x=prepare(x)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                if tta:
                    z=torch.stack([model(torch.rot90(x,k,[-2,-1])) for k in range(4)]).float().mean(0)
                else:z=model(x).float()
            out.append((z*ys+ym).cpu().numpy())
        return np.concatenate(out)
    protocol=dict(architecture='ContextFusion: shared ResNet18 at 224/896/1792 px; concatenation + LayerNorm + 256-unit GELU head',
      train=['A1','B1'],validation='C1',test='D1',target='fixed HEG200 panel-relative log1p, standardized per gene on train only',
      loss='standardized MSE + 0.2*(1-batch mean per-gene Pearson)',max_epochs=60,patience=12,
      seed=SEED,arms=ARMS,selection='C1 HEG200 PCC; best epoch then 1 vs 4 orientation TTA')
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    locked={};started=time.time()
    for arm in ARMS:
        ck=OUT/f'{arm}_best.pt';sel=OUT/f'{arm}_selection.json'
        if ck.exists() and sel.exists():
            locked[arm]=json.loads(sel.read_text());print('reuse completed',arm,flush=True);continue
        torch.manual_seed(SEED);np.random.seed(SEED);random.seed(SEED)
        model=ContextFusion(arm=='imagenet').cuda()
        opt=torch.optim.AdamW([dict(params=model.encoder.parameters(),lr=3e-5 if arm=='imagenet' else 3e-4),dict(params=model.head.parameters(),lr=3e-4)],weight_decay=.01)
        best=-1;stale=0;history=[]
        for epoch in range(1,61):
            model.train();losses=[]
            for x,y in tl:
                target=(y.cuda(non_blocking=True)-ym)/ys
                with torch.autocast('cuda',dtype=torch.bfloat16):z=model(prepare(x,True)).float()
                zc=z-z.mean(0);tc=target-target.mean(0)
                corr=(zc*tc).sum(0)/(zc.square().sum(0)*tc.square().sum(0)+1e-6).sqrt()
                loss=(z-target).square().mean()+.2*(1-corr.mean())
                opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
                losses.append(loss.item())
            pv=predict(model,vl);metric=common.score(val.y,pv)
            row=dict(epoch=epoch,train_loss=float(np.mean(losses)),validation=metric,elapsed_seconds=time.time()-started)
            history.append(row);print(arm,row,flush=True)
            if metric['HEG200']>best+1e-5:
                best=metric['HEG200'];stale=0
                torch.save(dict(model=model.state_dict(),epoch=epoch,ym=ym.cpu(),ys=ys.cpu()),ck)
            else:stale+=1
            (OUT/f'{arm}_training.json').write_text(json.dumps(history,indent=2))
            if stale>=12:break
        checkpoint=torch.load(ck,map_location='cuda',weights_only=True);model.load_state_dict(checkpoint['model'])
        candidates={str(t):common.score(val.y,predict(model,vl,t)) for t in [False,True]}
        tta=max([False,True],key=lambda t:candidates[str(t)]['HEG200'])
        locked[arm]=dict(epoch=checkpoint['epoch'],tta=tta,validation=candidates[str(tta)],candidates=candidates)
        (OUT/f'{arm}_selection.json').write_text(json.dumps(locked[arm],indent=2))
        np.savez_compressed(OUT/f'{arm}_C1.npz',truth=val.y,predicted=predict(model,vl,tta),genes=val.arrays[0]['genes'])
        del model,opt;torch.cuda.empty_cache()
    selected=max(locked,key=lambda arm:locked[arm]['validation']['HEG200'])
    (OUT/'selection_locked.json').write_text(json.dumps(dict(selected=selected,arms=locked,unix=time.time()),indent=2))
    test=Slides(['D1']);testloader=DataLoader(test,batch_size=32,num_workers=2,pin_memory=True)
    results={}
    for arm in locked:
        model=ContextFusion(False).cuda();model.load_state_dict(torch.load(OUT/f'{arm}_best.pt',map_location='cuda',weights_only=True)['model'])
        pred=predict(model,testloader,locked[arm]['tta'])
        results[arm]=common.score(test.y,pred)
        np.savez_compressed(OUT/f'{arm}_D1.npz',truth=test.y,predicted=pred,genes=test.arrays[0]['genes'],coordinates_xy=test.arrays[0]['coordinates_xy'],barcodes=test.arrays[0]['barcodes'])
        del model
    (OUT/'test_metrics.json').write_text(json.dumps(dict(selected=selected,arms=results),indent=2))
    print('FINAL',selected,results,flush=True)

if __name__=='__main__':main()
