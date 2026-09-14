"""BLEEP 单因素改进：用最终检索任务的 C1 PCC 选 epoch，不用对比损失代替。

保留冻结 ResNet18、官方投影头/对比损失与原训练标签，不修改作者仓库。
CPU 运行，避免与 GPU 上的 Stem/DenseNet 争用显存。
"""
from pathlib import Path
import importlib.util
import json
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results/improvement_20260908/bleep"
OUT.mkdir(parents=True,exist_ok=True)
spec=importlib.util.spec_from_file_location("baseline",ROOT/"scripts/07_train_image_baselines.py")
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42)


def normalize(x):
    x=np.maximum(x.astype(float),0)
    return np.log1p(10000*x/np.maximum(x.sum(1,keepdims=True),1e-12))


def score(truth,pred):
    truth=normalize(truth);pred=normalize(np.expm1(pred.astype(float)))
    a,c=truth-truth.mean(0),pred-pred.mean(0)
    corr=(a*c).sum(0)/np.maximum(np.sqrt((a*a).sum(0)*(c*c).sum(0)),1e-12)
    return {"pcc":float(corr.mean()),"mae":float(abs(truth-pred).mean())}


def main():
    train=[b.load_slide(s) for s in ["C73_A1","C73_B1"]]
    val=b.load_slide("C73_C1")
    x=np.concatenate([d["features"] for d in train]).astype(np.float32)
    y=np.concatenate([d["log_normalized"] for d in train]).astype(np.float32)
    mean,std=x.mean(0),np.maximum(x.std(0),1e-6)
    x=(x-mean)/std;vx=(val["features"]-mean)/std
    model=b.FrozenBLEEP(200)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    loader=DataLoader(TensorDataset(torch.from_numpy(x),torch.from_numpy(y)),batch_size=128,shuffle=True)
    best=-float("inf");stale=0;history=[];start=time.perf_counter()
    for epoch in range(1,121):
        model.train()
        for bx,by in loader:
            optimizer.zero_grad(set_to_none=True);loss=model(bx,by);loss.backward();optimizer.step()
        if epoch!=1 and epoch%10:continue
        _,reference=b.bleeper_embeddings(model,x,y,torch.device("cpu"))
        query,_=b.bleeper_embeddings(model,vx,val["log_normalized"],torch.device("cpu"))
        candidates={}
        for k in [10,50,100]:
            candidates[k]=score(val["raw_counts"],b.retrieve_expression(query,reference,y,k))
        k=max(candidates,key=lambda n:candidates[n]["pcc"])
        record={"epoch":epoch,"candidates":candidates,"elapsed":time.perf_counter()-start}
        history.append(record);print(record,flush=True)
        if candidates[k]["pcc"]>best+1e-5:
            best=candidates[k]["pcc"];stale=0
            torch.save(model.state_dict(),OUT/"best.pt")
            selection={"epoch":epoch,"neighbors":k,"validation":candidates[k],"selection":"full C1 panel PCC; no D1 involved"}
            (OUT/"selection.json").write_text(json.dumps(selection,indent=2))
        else:stale+=1
        (OUT/"training.json").write_text(json.dumps(history,indent=2))
        if stale>=4:break
    model.load_state_dict(torch.load(OUT/"best.pt",weights_only=True))
    _,reference=b.bleeper_embeddings(model,x,y,torch.device("cpu"))
    for slide in ["C73_C1","C73_D1"]:
        data=val if slide=="C73_C1" else b.load_slide(slide)
        query,_=b.bleeper_embeddings(model,(data["features"]-mean)/std,data["log_normalized"],torch.device("cpu"))
        prediction=b.retrieve_expression(query,reference,y,selection["neighbors"])
        np.savez_compressed(OUT/f"{slide}_predictions.npz",predicted=prediction,true=data["log_normalized"],coordinates_xy=data["coordinates_xy"],genes=data["genes"])
        result=score(data["raw_counts"],prediction)
        (OUT/f"{slide}_metrics.json").write_text(json.dumps(result,indent=2))
        print(slide,result,flush=True)


if __name__=="__main__":main()
