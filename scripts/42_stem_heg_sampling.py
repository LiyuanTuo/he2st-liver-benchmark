"""HEG 面板：Stem 采样消融（scripts/28 的 HEG 适配版：换数据、权重与输出目录）。

运行：wsl python3 scripts/42_stem_heg_sampling.py
只改变权重分支与采样约束，不更改基因、测试集和计分单位。
"""

import json
import sys
import time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/Stem"))
from Stem.diffusion import create_diffusion
from Stem.models import Stem_models

OUT = ROOT / "results/improvement_heg/stem"
OUT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(6)
torch.manual_seed(42)
torch.backends.cuda.matmul.allow_tf32 = True


def load(slide):
    with np.load(ROOT / f"data/processed/gse240429_heg/arrays/{slide}.npz") as d:
        data = {k: d[k] for k in d.files}
    data["features"] = np.load(ROOT / f"data/processed/gse240429/image/{slide}_resnet18.npy")
    return data


def normalize(x):
    x = np.maximum(x.astype(np.float64), 0)
    return np.log1p(10000 * x / np.maximum(x.sum(1, keepdims=True), 1e-12))


def score(raw, prediction):
    truth, pred = normalize(raw), normalize(np.expm1(np.minimum(prediction, 30)))
    a, b = truth-truth.mean(0), pred-pred.mean(0)
    den = np.sqrt((a*a).sum(0)*(b*b).sum(0))
    valid = den > 1e-12
    corr = np.divide((a*b).sum(0), den, out=np.zeros(200), where=valid)
    return {"pcc": float(corr.mean()), "mae": float(np.abs(truth-pred).mean()),
            "undefined_pcc": int((~valid).sum()), "negative_fraction": float((prediction<0).mean()),
            "raw_min": float(prediction.min()), "raw_max": float(prediction.max())}


def save_json(name, data):
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    train = [load(s) for s in ["C73_A1", "C73_B1"]]
    features = np.concatenate([d["features"] for d in train])
    expression = np.concatenate([d["log_normalized"] for d in train])
    mean, std = features.mean(0), np.maximum(features.std(0), 1e-6)
    upper = float(expression.max())
    checkpoint = torch.load(ROOT / "results/stem_adapted_heg/best.pt", map_location="cpu", weights_only=True)
    model = Stem_models["Stem"](input_size=200, depth=12, hidden_size=384, num_heads=6, label_size=512).cuda().eval()
    step_count = checkpoint.get("updates", checkpoint["epoch"] * (len(features)//32))
    diagnostics = {"epoch": checkpoint["epoch"], "estimated_updates": step_count,
                   "ema_decay": checkpoint.get("ema_decay", 0.9999),
                   "initial_ema_weight_fraction": checkpoint.get("ema_decay", 0.9999)**step_count,
                   "training_target_above_one": float((expression>1).mean()), "training_max": upper,
                   "selection": "C1 only; 256 fixed seed20260908 spots, 1 sample, DDIM50; full C1 audit; D1 evaluated only after selection"}
    save_json("diagnostics.json", diagnostics)
    print(diagnostics, flush=True)

    @torch.inference_mode()
    def sample(data, branch, clipping, steps, repeats):
        model.load_state_dict(checkpoint[branch])
        conditions = torch.from_numpy((data["features"]-mean)/std).float().cuda()
        diffusion = create_diffusion(str(steps))
        outputs = []
        for repeat in range(repeats):
            torch.manual_seed(20260908+repeat)
            rows = []
            for start in range(0, len(conditions), 64):
                y = conditions[start:start+64]
                z = torch.randn(len(y), 1, 200, device="cuda")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = diffusion.ddim_sample_loop(model.forward, z.shape, z,
                        clip_denoised=(clipping=="image_range"),
                        denoised_fn=(lambda x: x.clamp(0, upper)) if clipping=="training_range" else None,
                        model_kwargs={"y": y}, device="cuda", eta=0)
                rows.append(out[:, 0].float().cpu().numpy())
            outputs.append(np.concatenate(rows))
            print(f"sample {branch}/{clipping} {repeat+1}/{repeats} n={len(conditions)}", flush=True)
        return np.mean(outputs, axis=0)

    validation = load("C73_C1")
    indices = np.sort(np.random.default_rng(20260908).choice(len(validation["features"]), 256, replace=False))
    subset = {k: v[indices] for k, v in validation.items() if k != "genes"}
    trials = []
    for branch, clipping in [("ema", "image_range"), ("ema", "training_range"),
                             ("model", "image_range"), ("model", "training_range"),
                             ("model", "none")]:
        start = time.perf_counter()
        prediction = sample(subset, branch, clipping, 50, 1)
        metrics = score(subset["raw_counts"], prediction)
        trials.append({"branch": branch, "clipping": clipping,
                       "validation_subset": metrics, "seconds": time.perf_counter()-start})
        save_json("ablation.json", trials)
        print(trials[-1], flush=True)
    selected = max(trials, key=lambda t: t["validation_subset"]["pcc"])
    save_json("selection.json", selected)
    for slide in ["C73_C1", "C73_D1"]:
        data = validation if slide == "C73_C1" else load(slide)
        prediction = sample(data, selected["branch"], selected["clipping"], 100, 3)
        np.savez_compressed(OUT/f"{slide}_predictions.npz", predicted=prediction,
                            true=data["log_normalized"],
                            coordinates_xy=data["coordinates_xy"], genes=data["genes"])
        metrics = score(data["raw_counts"], prediction)
        save_json(f"{slide}_metrics.json", metrics)
        print(slide, metrics, flush=True)


if __name__ == "__main__":
    main()
