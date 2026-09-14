import sys
import time

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
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scaler = torch.amp.GradScaler("cuda")

x = torch.randn(128, 1, 200, device=device)
y = torch.randn(128, 512, device=device)

# AMP 前向
t = torch.randint(0, 1000, (128,), device=device)
torch.cuda.synchronize()
t0 = time.perf_counter()
with torch.autocast("cuda", dtype=torch.float16):
    out = model(x, t, y)
torch.cuda.synchronize()
print(f"AMP forward: {time.perf_counter()-t0:.3f}s")

# AMP 完整步 ×5
t0 = time.perf_counter()
for _ in range(5):
    tb = torch.randint(0, 1000, (128,), device=device)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.float16):
        loss = diffusion.training_losses(model, x, tb, dict(y=y))["loss"].mean()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
torch.cuda.synchronize()
print(f"5 AMP full steps: {time.perf_counter()-t0:.3f}s")
print(
    "VRAM used:",
    round(torch.cuda.memory_allocated() / 1e9, 2),
    "GB allocated,",
    round(torch.cuda.max_memory_allocated() / 1e9, 2),
    "GB peak",
)
