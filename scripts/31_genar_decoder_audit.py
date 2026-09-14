"""GenAR 解码诊断：保持粗尺度 top1，仅在最终尺度比较众数与概率期望。

复用官方推理和已有权重。先在完整 C1 固定解码，再运行 D1。
这是推理后处理实验，不宣称修复全部生成模型或完成原论文训练。
"""
import sys
import json
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"third_party/GenAR/src"))
from model import ModelInterface
OUT=ROOT/"results/improvement_20260908/genar"
OUT.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(4)


def norm(x):
    x=np.maximum(x.astype(float),0)
    return np.log1p(10000*x/np.maximum(x.sum(1,keepdims=True),1e-12))


def score(raw,pred):
    a,b=norm(raw),norm(pred)
    x,y=a-a.mean(0),b-b.mean(0)
    pcc=(x*y).sum(0)/np.maximum(np.sqrt((x*x).sum(0)*(y*y).sum(0)),1e-12)
    return {"pcc":float(pcc.mean()),"mae":float(abs(a-b).mean())}


def main():
    paths=list((ROOT/"logs/gse240429/GENAR").glob("*/*.ckpt"))
    assert len(paths)==1, "需要唯一原权重，不自动按测试分数挑 checkpoint"
    ckpt=torch.load(paths[0],map_location="cpu",weights_only=True)
    interface=ModelInterface(ckpt["hyper_parameters"]["config"])
    interface.load_state_dict(ckpt["state_dict"]);interface.eval()
    model=interface.model
    captured={}
    def hook(module,args,output):
        if output.shape[1]!=200:return
        # 在官方 top1 原地掩码之前读取 logits；不修改模型的输出。
        for k in [2,8,32]:
            values,indices=output.float().topk(k,dim=-1)
            captured[str(k)]=(values.softmax(-1)*indices).sum(-1).detach().numpy()
        captured["1"]=output.argmax(-1).float().detach().numpy()
    handle=model.output_head.register_forward_hook(hook)
    selection=None
    for slide in ["C73_C1","C73_D1"]:
        with np.load(ROOT/f"data/processed/gse240429/arrays/{slide}.npz") as d:
            data={k:d[k] for k in d.files}
        x=np.load(ROOT/f"data/processed/gse240429/image/{slide}_resnet18.npy")
        coords=data["coordinates_xy"]
        coords=(coords-coords.min(0))/(coords.max(0)-coords.min(0))
        outputs={str(k):[] for k in [1,2,8,32]}
        with torch.inference_mode():
            for start in range(0,len(x),16):
                model.inference(torch.from_numpy(x[start:start+16]),torch.from_numpy(coords[start:start+16]),top_k=1,seed=2021)
                for k in outputs:outputs[k].append(captured[k].copy())
                if start%512==0:print(slide,start,len(x),flush=True)
        outputs={k:np.concatenate(v) for k,v in outputs.items()}
        if slide=="C73_C1":
            scores={k:score(data["raw_counts"],v) for k,v in outputs.items()}
            k=max(scores,key=lambda key:scores[key]["pcc"])
            selection={"top_k_expectation":k,"validation_candidates":scores,
                       "checkpoint":str(paths[0].relative_to(ROOT)),"selection":"full C1 only; previous scales always top1"}
            (OUT/"selection.json").write_text(json.dumps(selection,indent=2))
            print(selection,flush=True)
        predicted=outputs[selection["top_k_expectation"]]
        # 保存计数/相对丰度；不伪装成整数离散采样。
        np.savez_compressed(OUT/f"{slide}_predictions.npz",predicted_counts=predicted,true_counts=data["raw_counts"],gene_names=data["genes"],coordinates_xy=data["coordinates_xy"])
        result=score(data["raw_counts"],predicted)
        (OUT/f"{slide}_metrics.json").write_text(json.dumps(result,indent=2))
        print(slide,result,flush=True)
        if slide=="C73_D1":
            old=np.load(ROOT/"results/genar_gse240429/C73_D1_predictions.npz")
            check={"top1_equals_original":bool(np.array_equal(old["predicted_counts"],outputs["1"])),
                   "max_top1_count_difference":float(abs(old["predicted_counts"]-outputs["1"]).max())}
            (OUT/"baseline_check.json").write_text(json.dumps(check,indent=2))
    handle.remove()


if __name__=="__main__":main()
