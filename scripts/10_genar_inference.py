"""GenAR 统一协议适配：测试切片推理（不改动 third_party/GenAR 代码）。

复用官方 `inference.main()`；只做两件事：
1. 把 `gse240429` 注入 `configs.DATASETS`；
2. 把固定参数压进 `sys.argv`，保证每次调用完全一致。

输出默认写到 results/genar_gse240429/<slide>/。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/GenAR/src"))

import configs
import inference as genar_inference

configs.DATASETS["gse240429"] = {
    "dir_name": "gse240429",
    "val_slides": "C73_C1",
    "test_slides": "C73_D1",
    "recommended_encoder": "resnet18",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-path", required=True)
    parser.add_argument("--slide-id", default="C73_D1")
    parser.add_argument("--data-root", default=str(ROOT / "data/processed"))
    parser.add_argument("--encoder", default="resnet18")
    parser.add_argument("--output-dir", default=str(ROOT / "results/genar_gse240429"))
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-gene-count", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2021)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sys.argv = [
        "genar_inference",
        "--ckpt-path", args.ckpt_path,
        "--dataset", "gse240429",
        "--slide-id", args.slide_id,
        "--data-root", args.data_root,
        "--encoder", args.encoder,
        "--output-dir", args.output_dir,
        "--gpu-id", str(args.gpu_id),
        "--batch-size", str(args.batch_size),
        "--max-gene-count", str(args.max_gene_count),
        "--seed", str(args.seed),
        "--save-predictions",
    ]
    raise SystemExit(genar_inference.main())
