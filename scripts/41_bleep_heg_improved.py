"""HEG 面板：BLEEP C1 选择版（复用 scripts/30，仅换数据与输出目录，CPU 运行）。

运行：wsl python3 scripts/41_bleep_heg_improved.py
"""

from pathlib import Path
import importlib.util
import sys

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "orig30", ROOT / "scripts/30_bleep_validation_selection.py"
)
mod = importlib.util.module_from_spec(spec)
sys.modules["orig30"] = mod
spec.loader.exec_module(mod)

mod.OUT = ROOT / "results/improvement_heg/bleep"
mod.OUT.mkdir(parents=True, exist_ok=True)
mod.b.ARRAYS = ROOT / "data/processed/gse240429_heg/arrays"
# mod.b.IMAGES 不变（ResNet18 特征与基因面板无关）。

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
mod.main()
