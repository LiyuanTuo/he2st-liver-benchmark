import sys

sys.path.insert(0, "third_party/GenAR/src")

# GenAR ModelInterface 导入 + 冒烟（无需 CUDA）
from model import ModelInterface  # noqa: E402

print("GenAR ModelInterface import OK")

# Stem 模型前向冒烟（CPU）
sys.path.insert(0, "third_party/Stem")
import torch  # noqa: E402

from Stem.models import Stem_models  # noqa: E402
from Stem.diffusion import create_diffusion  # noqa: E402

model = Stem_models["Stem"](input_size=200, depth=2, hidden_size=64, num_heads=4, label_size=512)
x = torch.randn(2, 1, 200)
t = torch.randint(0, 1000, (2,))
y = torch.randn(2, 512)
out = model(x, t, y)
print("Stem forward OK:", tuple(out.shape))

diffusion = create_diffusion("")
loss = diffusion.training_losses(model, x, t, dict(y=y))["loss"]
print("Stem diffusion training loss OK:", float(loss.mean()))
