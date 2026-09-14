import sys
import time

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

true = np.load("data/processed/gse240429/arrays/C73_D1.npz")["log_normalized"][:32]
print("true stats:", round(float(true.min()), 2), round(float(true.max()), 2),
      "mean", round(float(true.mean()), 3))

y = torch.randn(32, 512, device=device)
autocast = torch.autocast("cuda", dtype=torch.float16)


def sample(diffusion, clip, eta=None):
    z = torch.randn(32, 1, 200, device=device)
    with torch.inference_mode(), autocast:
        if eta is None:
            out = diffusion.p_sample_loop(
                model.forward, z.shape, z, clip_denoised=clip,
                model_kwargs=dict(y=y), device=device)
        else:
            out = diffusion.ddim_sample_loop(
                model.forward, z.shape, z, clip_denoised=clip,
                model_kwargs=dict(y=y), device=device, eta=eta)
    x = out[:, 0, :].cpu().numpy()
    print(f"  min {x.min():.2f} max {x.max():.2f} mean {x.mean():.3f} "
          f"std {x.std():.2f} neg {float((x < 0).mean()):.3f}")


for steps, tag, eta in [(50, "ddim50 eta=0 clip", 0.0),
                        (50, "ddim50 eta=1 clip", 1.0),
                        (100, "ddim100 eta=0 clip", 0.0),
                        (1000, "ddpm1000 clip", None)]:
    diffusion = create_diffusion(str(steps))
    print(tag)
    t0 = time.perf_counter()
    sample(diffusion, clip=True, eta=eta)
    print(f"  time {time.perf_counter()-t0:.1f}s")
