import sys

sys.path.insert(0, "third_party/Stem")

import torch

from Stem.diffusion import create_diffusion
from Stem.models import Stem_models

torch.manual_seed(42)
device = torch.device("cuda")
model = Stem_models["Stem"](
    input_size=200, depth=12, hidden_size=384, num_heads=6, label_size=512
).to(device)
model.train()
diffusion = create_diffusion("")

x = torch.randn(128, 1, 200, device=device)
y = torch.randn(128, 512, device=device)
t = torch.randint(0, 1000, (128,), device=device)


def mem(tag):
    torch.cuda.synchronize()
    print(f"{tag:24s} allocated={torch.cuda.memory_allocated()/1e9:.2f}GB "
          f"peak={torch.cuda.max_memory_allocated()/1e9:.2f}GB")


mem("after init")
with torch.autocast("cuda", dtype=torch.float16):
    out = model(x, t, y)
mem("after forward")
with torch.autocast("cuda", dtype=torch.float16):
    terms = diffusion.training_losses(model, x, t, dict(y=y))
mem("after loss (no backward)")
del terms, out
torch.cuda.empty_cache()
mem("after del + empty_cache")

# 只算 MSE 部分（不进入 VB 路径）
with torch.autocast("cuda", dtype=torch.float16):
    model_output = model(x, t, y)
    mean_out, var_out = torch.split(model_output, 1, dim=1)
noise = torch.randn_like(x)
import math

xt = diffusion.q_sample(x, t, noise=noise)
with torch.autocast("cuda", dtype=torch.float16):
    model_output = model(xt, t, y)
    mean_out, var_out = torch.split(model_output, 1, dim=1)
mse = ((noise - mean_out) ** 2).mean()
mem("after mse graph")
loss = mse
loss.backward()
mem("after backward")
