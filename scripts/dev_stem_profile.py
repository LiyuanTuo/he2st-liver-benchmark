import sys

sys.path.insert(0, "third_party/Stem")

import torch

from Stem.models import Stem_models

torch.manual_seed(42)
device = torch.device("cuda")
model = Stem_models["Stem"](
    input_size=200, depth=12, hidden_size=384, num_heads=6, label_size=512
).to(device)
model.train()

x = torch.randn(128, 1, 200, device=device)
t = torch.randint(0, 1000, (128,), device=device)
y = torch.randn(128, 512, device=device)

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
) as prof:
    out = model(x, t, y)
    torch.cuda.synchronize()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
