"""GenAR 统一协议适配：训练（不改动 third_party/GenAR 代码）。

通过运行时向 `configs.DATASETS` 注入 `gse240429` 条目，复用官方
`main.build_config_from_args + main.main` 的完整流程。数据布局：

    data/processed/gse240429/
        st/*.h5ad                     原始整数计数 + obsm['spatial']
        processed_data/
            all_slide_lst.txt
            selected_gene_list.txt     200 基因层次顺序（SHA256 已记录）
            spot_features_resnet18/    与 h5ad 行序一致的特征张量

统一协议：train=C73_A1,C73_B1 / val=C73_C1 / test=C73_D1，
encoder=resnet18（本项目统一图像编码器），这是“统一数据上的适配实现”，
不是论文原表格数值（论文用 UNI 特征 + H100）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/GenAR/src"))

import configs
import main as genar_main

configs.DATASETS["gse240429"] = {
    "dir_name": "gse240429",
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
    parser.add_argument(
        "--allow-nondeterministic", action="store_true", default=True,
        help="关闭 Lightning deterministic 模式（默认开：8GB 适配下 "
             "adaptive_avg_pool2d backward 无确定性实现，会直接报错）",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sys.argv = [
        "genar_main",
        "--dataset", "gse240429",
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
    print("Runtime config:", dict(runtime_config) if False else "")
    genar_main.main(runtime_config)
