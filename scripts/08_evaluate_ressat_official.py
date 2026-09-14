"""评测 ResSAT 官方示例，并量化其预测对 batch 大小的敏感性。

官方自注意力把当前 DataLoader batch 当作 spot 序列，因此同一个 spot 的输出
可能随 batch size 和同批样本改变。本脚本在同一权重、同一测试顺序下分别用
1/8/16/32/64 推理；batch=32 是官方默认口径。
"""

from __future__ import annotations

import json
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party/ResSAT"))

from he2st.metrics import evaluate_normalized
from ressat.data_loader import load_gene_names, load_sections
from ressat.models import ResSAT
from ressat.utils import back_project


DATA = ROOT / "data/official_ressat_example"
RESULTS = ROOT / "results/ressat_official"
EXPERIMENT = "official_seed42"
BATCH_SIZES = (1, 8, 16, 32, 64)


def clean_json(value):
    if isinstance(value, dict):
        return {key: clean_json(item) for key, item in value.items()}
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def as_uint8_patches(sections):
    """把官方示例的 float32(0-255) patch 转成 uint8，避免 torchvision ToPILImage
    的 float→[0,1] 缩放造成数值回绕（与 scripts/06 的训练修复一致）。"""

    for section in sections:
        section["data"] = [
            (patch.astype(np.uint8, copy=False), embedding)
            for patch, embedding in section["data"]
        ]
    return sections


def main() -> None:
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    _, _, test = load_sections(str(DATA), section_num=2)
    test = as_uint8_patches(test)
    model = ResSAT(
        test_sections=test,
        data_dir=str(DATA),
        result_dir=str(RESULTS),
        patch_size=224,
        num_fourier=128,
        sigma=1,
        dropout=0.3,
        num_workers=4,
        exp_name=EXPERIMENT,
        gene_names=load_gene_names(str(DATA)),
    )
    model.load_checkpoint()
    with open(DATA / "pca_info.pkl", "rb") as source:
        pca_info = pickle.load(source)
    coordinates = np.asarray(test[0]["locs"], dtype=np.float32)

    predictions = {}
    metrics = {}
    for batch_size in BATCH_SIZES:
        predicted_pca, true_pca = model.predict(batch_size=batch_size)
        predicted = back_project(predicted_pca, pca_info).numpy()
        true = back_project(true_pca, pca_info).numpy()
        predictions[batch_size] = predicted
        result, _ = evaluate_normalized(true, predicted, coordinates)
        # 官方还报告按测试真值平均表达选出的 50 个高表达基因。
        high_expression = np.argsort(true.mean(axis=0))[-50:]
        correlations = [
            np.corrcoef(true[:, index], predicted[:, index])[0, 1]
            for index in high_expression
        ]
        result["official_top50_high_expression_pcc"] = float(np.nanmean(correlations))
        metrics[str(batch_size)] = result
        print(f"batch={batch_size}: PCC={result['pcc_macro_all_fixed_genes']:.4f}")

    reference = predictions[32]
    stability = {}
    for batch_size, predicted in predictions.items():
        difference = predicted - reference
        stability[str(batch_size)] = {
            "mean_absolute_difference_vs_batch32": float(np.abs(difference).mean()),
            "root_mean_squared_difference_vs_batch32": float(np.sqrt(np.mean(difference**2))),
            "maximum_absolute_difference_vs_batch32": float(np.abs(difference).max()),
            "flattened_prediction_correlation_vs_batch32": float(
                np.corrcoef(predicted.ravel(), reference.ravel())[0, 1]
            ),
        }

    np.savez_compressed(
        RESULTS / "official_batch32_predictions.npz",
        predicted=reference.astype(np.float32),
        true=true.astype(np.float32),
        coordinates_xy=coordinates,
        genes=np.asarray(pca_info["gene_names"]),
    )
    output = {
        "protocol": {
            "source": "ResSAT official Zenodo processed example",
            "train_section": "Section_2 (2251 spots)",
            "test_section": "Section_1 (2459 spots)",
            "checkpoint_selection": "minimum training loss; official two-section workflow has no validation",
            "default_inference_batch_size": 32,
        },
        "metrics_by_inference_batch_size": metrics,
        "batch_size_stability": stability,
    }
    (RESULTS / "evaluation.json").write_text(
        json.dumps(clean_json(output), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
