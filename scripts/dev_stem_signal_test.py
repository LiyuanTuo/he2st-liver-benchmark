import sys

sys.path.insert(0, "third_party/Stem")

import numpy as np
import torch

from Stem.diffusion import create_diffusion
from Stem.models import Stem_models

torch.manual_seed(42)
device = torch.device("cuda")
model = Stem_models["Stem"](
    input_size=200, depth=12, hidden_size=384, num_heads=6, label_size=512
)
ckpt = torch.load("results/stem_adapted/best.pt", map_location="cpu")
model.load_state_dict(ckpt["ema"])
model.to(device)
model.eval()

arrays = np.load("data/processed/gse240429/arrays/C73_D1.npz")
true = arrays["log_normalized"][:128].astype(np.float64)
features = np.load("data/processed/gse240429/image/C73_D1_resnet18.npy", mmap_mode="r")[:128].copy()
train_features = np.concatenate([
    np.load(f"data/processed/gse240429/image/{s}_resnet18.npy", mmap_mode="r").copy()
    for s in ("C73_A1", "C73_B1")
])
fm, fs = train_features.mean(0, keepdims=True), np.maximum(train_features.std(0, keepdims=True), 1e-6)
y = torch.from_numpy(((features - fm) / fs).astype(np.float32)).to(device)
autocast = torch.autocast("cuda", dtype=torch.float16)

results = {}
for steps, tag, eta in [(100, "ddim100_eta0_clip", 0.0),
                        (1000, "ddpm1000_clip", None),
                        (1000, "ddpm1000_noclip", "nc")]:
    diffusion = create_diffusion(str(steps))
    z = torch.randn(128, 1, 200, device=device)
    with torch.inference_mode(), autocast:
        if eta == "nc":
            out = diffusion.p_sample_loop(
                model.forward, z.shape, z, clip_denoised=False,
                model_kwargs=dict(y=y), device=device)
        elif eta is None:
            out = diffusion.p_sample_loop(
                model.forward, z.shape, z, clip_denoised=True,
                model_kwargs=dict(y=y), device=device)
        else:
            out = diffusion.ddim_sample_loop(
                model.forward, z.shape, z, clip_denoised=True,
                model_kwargs=dict(y=y), device=device, eta=eta)
    pred = out[:, 0, :].cpu().numpy().astype(np.float64)
    pccs = [np.corrcoef(true[:, g], pred[:, g])[0, 1] for g in range(200)]
    pccs = [p for p in pccs if np.isfinite(p)]
    print(f"{tag:22s} macro PCC={np.nanmean(pccs):+.4f} "
          f"spotcos={np.mean([np.dot(true[i], pred[i])/(np.linalg.norm(true[i])*np.linalg.norm(pred[i])+1e-9) for i in range(128)]):.4f} "
          f"mean={pred.mean():.3f} std={pred.std():.2f}")
