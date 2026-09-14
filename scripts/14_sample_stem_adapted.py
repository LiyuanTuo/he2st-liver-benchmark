"""Stem 统一协议适配：测试切片采样与评测（复用官方扩散采样代码）。

- 加载 scripts/13 训练的 EMA 权重；
- 用 250 步 respaced 扩散为每个测试 spot 采样 S=5 个表达样本；
- 以样本均值作为点预测，用统一指标评测；同时报告逐基因采样方差
  （Stem 的核心卖点：表达不确定性）。
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party/Stem"))

from he2st.metrics import evaluate_normalized
from Stem.diffusion import create_diffusion
from Stem.models import Stem_models


ARRAYS = ROOT / "data/processed/gse240429/arrays"
IMAGES = ROOT / "data/processed/gse240429/image"
OUTPUT = ROOT / "results/stem_adapted"
TEST_SLIDE = "C73_D1"

SAMPLES_PER_SPOT = 3
SAMPLING_STEPS = 100
SAMPLING_BATCH = 128
CLIP_DENOISED = True  # 必要：欠训练模型的高噪声步 x0 估计会爆炸（±900），必须裁剪
HIDDEN_SIZE = 384
DEPTH = 12
NUM_HEADS = 6
SEED = 42


def json_ready(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> None:
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with np.load(ARRAYS / f"{TEST_SLIDE}.npz") as source:
        true = source["log_normalized"].astype(np.float32)
        coordinates = source["coordinates_xy"].astype(np.float32)
        genes = source["genes"]
    features = np.load(IMAGES / f"{TEST_SLIDE}_resnet18.npy", mmap_mode="r").copy()

    train_features = np.concatenate(
        [
            np.load(IMAGES / f"{slide}_resnet18.npy", mmap_mode="r").copy()
            for slide in ("C73_A1", "C73_B1")
        ]
    )
    feature_mean = train_features.mean(axis=0, keepdims=True)
    feature_std = np.maximum(train_features.std(axis=0, keepdims=True), 1e-6)
    conditions = (features - feature_mean) / feature_std

    model = Stem_models["Stem"](
        input_size=true.shape[1],
        depth=DEPTH,
        hidden_size=HIDDEN_SIZE,
        num_heads=NUM_HEADS,
        label_size=conditions.shape[1],
    )
    checkpoint = torch.load(OUTPUT / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["ema"])
    model.to(device)
    model.eval()
    diffusion = create_diffusion(str(SAMPLING_STEPS))

    all_samples = []
    autocast = torch.autocast("cuda", dtype=torch.float16)
    spots = len(conditions)
    repeated = np.empty((SAMPLES_PER_SPOT, spots, true.shape[1]), dtype=np.float32)
    with torch.inference_mode():
        for sample_index in range(SAMPLES_PER_SPOT):
            for start in range(0, spots, SAMPLING_BATCH):
                stop = min(start + SAMPLING_BATCH, spots)
                batch_conditions = torch.from_numpy(
                    conditions[start:stop]
                ).float().to(device)
                z = torch.randn(
                    batch_conditions.shape[0], 1, true.shape[1], device=device
                )
                with autocast:
                    out = diffusion.ddim_sample_loop(
                        model.forward,
                        z.shape,
                        z,
                        clip_denoised=CLIP_DENOISED,
                        model_kwargs=dict(y=batch_conditions),
                        device=device,
                        eta=0.0,
                    )
                repeated[sample_index, start:stop] = out[:, 0, :].cpu().numpy()
            print(f"  sample {sample_index + 1}/{SAMPLES_PER_SPOT} 完成")

    mean_prediction = repeated.mean(axis=0)
    metrics, per_gene_pcc = evaluate_normalized(true, mean_prediction, coordinates)
    per_gene_sample_std = repeated.std(axis=0).mean(axis=0)
    per_gene_true_std = true.std(axis=0)
    finite = np.isfinite(per_gene_true_std) & (per_gene_true_std > 1e-8)
    rvd = float(
        np.mean(
            np.abs(
                per_gene_sample_std[finite] / per_gene_true_std[finite] - 1.0
            )
        )
    )

    np.savez_compressed(
        OUTPUT / "stem_samples_D1.npz",
        samples=repeated,
        mean_prediction=mean_prediction,
        true=true,
        coordinates_xy=coordinates,
        genes=genes,
    )
    summary = {
        "protocol": {
            "test_slide": TEST_SLIDE,
            "checkpoint": "scripts/13 best.pt (EMA)",
            "samples_per_spot": SAMPLES_PER_SPOT,
            "sampling_steps": SAMPLING_STEPS,
            "sampler": "DDIM eta=0, clip_denoised=True",
            "seed": SEED,
            "note": "mean of 3 samples used as point prediction; "
                    "per-gene sample std vs true std reported as RVD(samples). "
                    "clip_denoised=True is required: the adapted model's high-noise-step "
                    "x0 estimates explode otherwise (measured ±900 without clip).",
        },
        "test_metrics": metrics,
        "per_gene_sample_vs_true_rvd": rvd,
    }
    (OUTPUT / "sampling_evaluation.json").write_text(
        json.dumps(json_ready(summary), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(json_ready(metrics), indent=2, ensure_ascii=False))
    print(f"RVD(samples vs true std): {rvd:.4f}")
    print(f"完成：{OUTPUT / 'sampling_evaluation.json'}")


if __name__ == "__main__":
    main()
