"""HEG 面板：训练 GenAR 统一协议适配（与 scripts/09 相同流程，数据集换 gse240429_heg）。

运行：wsl python3 scripts/38_train_heg_genar.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/GenAR/src"))

import configs
import main as genar_main

configs.DATASETS["gse240429_heg"] = {
    "dir_name": "gse240429_heg",
    "val_slides": "C73_C1",
    "test_slides": "C73_D1",
    "recommended_encoder": "resnet18",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(ROOT / "data/processed"))
    parser.add_argument("--encoder", default="resnet18")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--global-batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--max-gene-count", type=int, default=2000)
    parser.add_argument("--precision", default="16-mixed")
    parser.add_argument("--allow-nondeterministic", action="store_true", default=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sys.argv = [
        "genar_main",
        "--dataset", "gse240429_heg",
        "--data-root", args.data_root,
        "--encoder", args.encoder,
        "--gpus", str(args.gpus),
        "--global-batch-size", str(args.global_batch_size),
        "--epochs", str(args.epochs),
        "--lr", str(args.lr),
        "--seed", str(args.seed),
        "--max-gene-count", str(args.max_gene_count),
        "--precision", args.precision,
    ]
    if args.allow_nondeterministic:
        sys.argv.append("--allow-nondeterministic")
    runtime_config = genar_main.build_config_from_args(
        genar_main.get_parse().parse_args()
    )
    genar_main.main(runtime_config)
