"""Stem 训练性能诊断：逐阶段计时，定位慢路径。"""

import sys
import time
from copy import deepcopy

sys.path.insert(0, "third_party/Stem")

import numpy as np
import torch

from Stem.diffusion import create_diffusion
from Stem.models import Stem_models

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

model = Stem_models["Stem"](
    input_size=200, depth=12, hidden_size=384, num_heads=6, label_size=512
).to(device)
print("params:", sum(p.numel() for p in model.parameters()) / 1e6, "M")
ema = deepcopy(model)
ema.eval()
for p in ema.parameters():
    p.requires_grad_(False)
diffusion = create_diffusion("")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)

x = torch.randn(128, 1, 200, device=device)
y = torch.randn(128, 512, device=device)
t = torch.randint(0, diffusion.num_timesteps, (128,), device=device)

# 单次前向
model.train()
start = time.perf_counter()
out = model(x, t, y)
torch.cuda.synchronize()
print(f"forward: {time.perf_counter() - start:.3f}s, out {tuple(out.shape)}")

# 单步损失（含 VB 路径）
start = time.perf_counter()
terms = diffusion.training_losses(model, x, t, dict(y=y))
loss = terms["loss"].mean()
torch.cuda.synchronize()
print(f"training_losses: {time.perf_counter() - start:.3f}s")

# 反向 + 优化器 + EMA
start = time.perf_counter()
optimizer.zero_grad(set_to_none=True)
loss.backward()
torch.cuda.synchronize()
print(f"backward: {time.perf_counter() - start:.3f}s")

start = time.perf_counter()
optimizer.step()
torch.cuda.synchronize()
print(f"optimizer.step: {time.perf_counter() - start:.3f}s")

start = time.perf_counter()
with torch.no_grad():
    for ema_param, param in zip(ema.parameters(), model.parameters()):
        ema_param.mul_(0.9999).add_(param.data, alpha=1e-4)
torch.cuda.synchronize()
print(f"ema: {time.perf_counter() - start:.3f}s")

# 完整 5 步（含 .to(device) 拷贝）
x_cpu = torch.randn(128, 200)
y_cpu = torch.randn(128, 512)
start = time.perf_counter()
for _ in range(5):
    xb = x_cpu.unsqueeze(1).to(device)
    yb = y_cpu.to(device)
    tb = torch.randint(0, diffusion.num_timesteps, (128,), device=device)
    optimizer.zero_grad(set_to_none=True)
    l = diffusion.training_losses(model, xb, tb, dict(y=yb))["loss"].mean()
    l.backward()
    optimizer.step()
torch.cuda.synchronize()
print(f"5 full steps: {time.perf_counter() - start:.3f}s")
