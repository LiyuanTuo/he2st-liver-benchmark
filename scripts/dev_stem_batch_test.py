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

for batch in (64, 32):
    x = torch.randn(batch, 1, 200, device=device)
    y = torch.randn(batch, 512, device=device)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(5):
        tb = torch.randint(0, 1000, (batch,), device=device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = diffusion.training_losses(model, x, tb, dict(y=y))["loss"].mean()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print(f"batch={batch}: {dt/5:.2f}s/step, peak={torch.cuda.max_memory_allocated()/1e9:.2f}GB")
