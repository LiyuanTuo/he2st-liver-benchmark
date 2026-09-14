"""HEG 面板：构造 ResSAT 统一数据集（复用 scripts/11，仅换输入/输出目录）。

运行：wsl python3 scripts/36_build_heg_ressat_data.py
"""

from pathlib import Path
import importlib.util
import sys

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "orig11", ROOT / "scripts/11_build_ressat_unified_data.py"
)
mod = importlib.util.module_from_spec(spec)
sys.modules["orig11"] = mod
spec.loader.exec_module(mod)

mod.ARRAYS = ROOT / "data/processed/gse240429_heg/arrays"
mod.OUTPUT = ROOT / "data/processed/ressat_gse240429_heg"
# mod.PATCHES 不变：图像块与基因面板无关，barcode 行序已断言一致。

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
mod.main()
