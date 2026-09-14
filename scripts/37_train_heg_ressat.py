"""HEG 面板：训练并评测 ResSAT 统一适配（复用 scripts/12，仅换数据/结果目录）。

运行：wsl python3 scripts/37_train_heg_ressat.py
"""

from pathlib import Path
import importlib.util
import sys

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "orig12", ROOT / "scripts/12_train_ressat_unified.py"
)
mod = importlib.util.module_from_spec(spec)
sys.modules["orig12"] = mod
spec.loader.exec_module(mod)

mod.DATA = ROOT / "data/processed/ressat_gse240429_heg"
mod.RESULTS = ROOT / "results/ressat_unified_heg"
mod.ARRAYS = ROOT / "data/processed/gse240429_heg/arrays"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
mod.main()
