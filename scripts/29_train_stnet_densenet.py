"""把原来的轻量 MLP 替换为 ST-Net 的 DenseNet121 + 多基因回归头。

这是统一人肝数据上的适配，不声称复现原论文乳腺队列和数值。
保留作者模型结构、零权重/训练均值偏置初始化、8种旋转镜像增强；
显式差异：200基因、AdamW、24轮预算、C1早停、固定共同评测口径。
运行：WSL python3 scripts/29_train_stnet_densenet.py
"""
from pathlib import Path
import importlib.util
import json
import random
import sys
import time
import types
import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import densenet121, DenseNet121_Weights

ROOT=Path(__file__).resolve().parents[1]
PANEL=os.environ.get("HE2ST_PANEL", "gse240429")
OUT=ROOT/("results/improvement_heg/stnet" if PANEL == "gse240429_heg" else "results/improvement_20260908/stnet")
OUT.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(6)
torch.manual_seed(42); np.random.seed(42); random.seed(42)
torch.backends.cudnn.benchmark=True
torch.backends.cuda.matmul.allow_tf32=True


class Patches(Dataset):
    def __init__(self,slides):
        self.patches=[np.load(ROOT/f"data/processed/gse240429/image/{s}_patches_uint8.npy",mmap_mode="r") for s in slides]
        self.arrays=[]
        for s in slides:
            with np.load(ROOT/f"data/processed/{PANEL}/arrays/{s}.npz") as d:
                self.arrays.append({k:d[k] for k in d.files})
        self.sizes=np.cumsum([len(p) for p in self.patches])
        self.true=np.concatenate([d["log_normalized"] for d in self.arrays])
        self.raw=np.concatenate([d["raw_counts"] for d in self.arrays])

    def __len__(self):
        return len(self.true)

    def __getitem__(self,i):
        slide=int(np.searchsorted(self.sizes,i,side="right"))
        local=i-(self.sizes[slide-1] if slide else 0)
        return torch.from_numpy(self.patches[slide][local].copy()).permute(2,0,1), torch.from_numpy(self.true[i])


def norm(x):
    x=np.maximum(x.astype(np.float64),0)
    return np.log1p(10000*x/np.maximum(x.sum(1,keepdims=True),1e-12))


def metrics(raw,predicted):
    true,pred=norm(raw),norm(np.expm1(predicted.astype(np.float64)))
    x,y=true-true.mean(0),pred-pred.mean(0)
    den=np.sqrt((x*x).sum(0)*(y*y).sum(0))
    pcc=np.divide((x*y).sum(0),den,out=np.zeros(200),where=den>1e-12)
    return {"pcc":float(pcc.mean()),"mae":float(np.abs(true-pred).mean()),"undefined":int((den<=1e-12).sum())}


def main():
    train=Patches(["C73_A1","C73_B1"])
    validation=Patches(["C73_C1"])
    train_loader=DataLoader(train,batch_size=32,shuffle=True,num_workers=2,pin_memory=True,persistent_workers=True)
    val_loader=DataLoader(validation,batch_size=64,num_workers=2,pin_memory=True,persistent_workers=True)
    model=densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
    # 仅加载作者的独立工具文件，绕开老仓库 import 时自动解析 CLI 的副作用。
    previous=sys.modules.get("stnet")
    sys.modules["stnet"]=types.ModuleType("stnet")
    spec=importlib.util.spec_from_file_location("stnet_original_nn",ROOT/"third_party/ST-Net/stnet/utils/nn.py")
    author=importlib.util.module_from_spec(spec); spec.loader.exec_module(author)
    if previous is None: del sys.modules["stnet"]
    else: sys.modules["stnet"]=previous
    author.set_out_features(model,200)
    model.classifier.bias.data.copy_(torch.from_numpy(train.true.mean(0)))
    model.cuda().to(memory_format=torch.channels_last)
    optimizer=torch.optim.AdamW([{"params":model.features.parameters(),"lr":3e-5},
                                {"params":model.classifier.parameters(),"lr":3e-4}],weight_decay=1e-4)
    center=torch.tensor([.485,.456,.406],device="cuda")[None,:,None,None]
    scale=torch.tensor([.229,.224,.225],device="cuda")[None,:,None,None]
    def prepare(images,augment=False):
        images=images.cuda(non_blocking=True).float().div_(255)
        if augment:
            # 四个旋转×镜像；batch随机增强，输出标签不随旋转改变。
            images=torch.rot90(images,random.randrange(4),[-2,-1])
            if random.random()<.5: images=images.flip(-1)
        return ((images-center)/scale).contiguous(memory_format=torch.channels_last)

    @torch.inference_mode()
    def predict(loader,tta=False):
        model.eval(); rows=[]
        for images,_ in loader:
            image=prepare(images)
            with torch.autocast("cuda",dtype=torch.bfloat16):
                if tta:
                    values=[model(torch.rot90(image,k,[-2,-1]).flip(-1) if mirror else torch.rot90(image,k,[-2,-1])) for mirror in [False,True] for k in range(4)]
                    prediction=torch.stack(values).float().mean(0)
                else: prediction=model(image).float()
            rows.append(prediction.cpu().numpy())
        return np.concatenate(rows)

    history=[]; best=-float("inf"); stale=0; started=time.perf_counter()
    for epoch in range(1,25):
        model.train(); losses=[]
        for images,target in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.bfloat16):
                prediction=model(prepare(images,True))
                loss=nn.functional.mse_loss(prediction.float(),target.cuda(non_blocking=True))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5); optimizer.step()
            losses.append(float(loss.detach()))
        val=metrics(validation.raw,predict(val_loader))
        record={"epoch":epoch,"train_mse":float(np.mean(losses)),"validation":val,"elapsed_seconds":time.perf_counter()-started}
        history.append(record); print(record,flush=True)
        if val["pcc"]>best+1e-5:
            best=val["pcc"];stale=0
            torch.save({"model":model.state_dict(),"epoch":epoch},OUT/"best.pt")
        else: stale+=1
        (OUT/"training.json").write_text(json.dumps(history,indent=2))
        if stale>=6:break
    checkpoint=torch.load(OUT/"best.pt",map_location="cuda",weights_only=True)
    model.load_state_dict(checkpoint["model"])
    # 是否使用论文的8方向TTA仍由C1选择，D1尚未载入。
    candidate={}
    for tta in [False,True]:
        candidate[str(tta)]=metrics(validation.raw,predict(val_loader,tta))
    tta=max([False,True],key=lambda t:candidate[str(t)]["pcc"])
    selection={"epoch":checkpoint["epoch"],"tta":tta,"validation_candidates":candidate,
               "model":"DenseNet121 + 200-gene linear head, official set_out_features reused",
               "training":"all weights finetuned; ImageNet; AdamW 3e-5 backbone/3e-4 head; MSE; rotation/mirror augmentation; max24 epochs; patience6 C1 PCC",
               "evaluation":"same fixed 200 genes, panel-relative log1p normalization as 9-page PPT"}
    (OUT/"selection.json").write_text(json.dumps(selection,indent=2))
    for slide,data in [("C73_C1",validation),("C73_D1",Patches(["C73_D1"]))]:
        loader=DataLoader(data,batch_size=64,num_workers=2,pin_memory=True)
        prediction=predict(loader,tta)
        np.savez_compressed(OUT/f"{slide}_predictions.npz",predicted=prediction,true=data.true,
                            coordinates_xy=data.arrays[0]["coordinates_xy"],genes=data.arrays[0]["genes"])
        result=metrics(data.raw,prediction)
        (OUT/f"{slide}_metrics.json").write_text(json.dumps(result,indent=2))
        print(slide,result,flush=True)


if __name__=="__main__":main()
