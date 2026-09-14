import sys
import time

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


def timed(tag, fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    print(f"{tag:30s} {time.perf_counter()-t0:.3f}s")
    return out


x_emb = timed("gene_joint_embed", lambda: model.gene_joint_embed(x))
t_emb = timed("time_embed", lambda: model.time_embed(t))
y_emb = timed("label_embed", lambda: model.label_embed(y))
c = t_emb + y_emb
for i, block in enumerate(model.blocks):
    x_emb = timed(f"block {i}", lambda x_emb=x_emb, block=block: block(x_emb, c))
out = timed("final_layer", lambda: model.final_layer(x_emb, c))
print("final out:", tuple(out.shape))
