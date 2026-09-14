"""HEG 面板：训练三种 H&E 图像基线（复用 scripts/07 全部逻辑，仅换数据/输出目录）。

运行：wsl python3 scripts/35_train_heg_image_baselines.py
"""

from pathlib import Path
import importlib.util
import sys

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "orig07", ROOT / "scripts/07_train_image_baselines.py"
)
mod = importlib.util.module_from_spec(spec)
sys.modules["orig07"] = mod
spec.loader.exec_module(mod)

mod.ARRAYS = ROOT / "data/processed/gse240429_heg/arrays"
mod.OUTPUT = ROOT / "results/unified_image_baselines_heg"
# mod.IMAGES 不变：ResNet18 特征与基因面板无关，直接复用旧缓存。

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
mod.main()
