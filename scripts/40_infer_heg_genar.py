"""HEG 面板：GenAR 测试切片推理（同 scripts/10 流程，数据集 gse240429_heg，自动找权重）。

运行：wsl python3 scripts/40_infer_heg_genar.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/GenAR/src"))

import configs
import inference as genar_inference

configs.DATASETS["gse240429_heg"] = {
    "dir_name": "gse240429_heg",
    "val_slides": "C73_C1",
    "test_slides": "C73_D1",
    "recommended_encoder": "resnet18",
}


def find_checkpoint():
    candidates = sorted(
        (ROOT / "logs/gse240429_heg/GENAR").glob("*/*.ckpt"),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError("未找到 GenAR HEG 权重：logs/gse240429_heg/GENAR/*/*.ckpt")
    return str(candidates[-1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-path", default=None)
    parser.add_argument("--slide-id", default="C73_D1")
    parser.add_argument("--data-root", default=str(ROOT / "data/processed"))
    parser.add_argument("--encoder", default="resnet18")
    parser.add_argument("--output-dir", default=str(ROOT / "results/genar_gse240429_heg"))
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-gene-count", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2021)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    checkpoint = args.ckpt_path or find_checkpoint()
    print(f"checkpoint: {checkpoint}", flush=True)
    sys.argv = [
        "genar_inference",
        "--ckpt-path", checkpoint,
        "--dataset", "gse240429_heg",
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
