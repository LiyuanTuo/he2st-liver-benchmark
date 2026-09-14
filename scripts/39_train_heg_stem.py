"""HEG 面板：训练 Stem 统一协议适配（复用 scripts/13，仅换数据/输出目录）。

运行：wsl python3 scripts/39_train_heg_stem.py
"""

from pathlib import Path
import importlib.util
import sys

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "orig13", ROOT / "scripts/13_train_stem_adapted.py"
)
mod = importlib.util.module_from_spec(spec)
sys.modules["orig13"] = mod
spec.loader.exec_module(mod)

mod.ARRAYS = ROOT / "data/processed/gse240429_heg/arrays"
mod.OUTPUT = ROOT / "results/stem_adapted_heg"
# mod.IMAGES 不变：ResNet18 特征与基因面板无关。

# Finite-budget adaptation: the previous EMA retained 64% of initialization
# at its selected checkpoint. A shorter averaging horizon follows training.
mod.EPOCHS = 100
mod.PATIENCE = 25
mod.EMA_DECAY = 0.99
mod.torch.set_num_threads(6)
mod.torch.backends.cuda.matmul.allow_tf32 = True

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
mod.main()
