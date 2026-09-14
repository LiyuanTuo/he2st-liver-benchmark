"""ResSAT 统一协议适配：在 GSE240429 上训练官方模型主体并评测。

数据由 scripts/11_build_ressat_unified_data.py 生成（patch 已存为 uint8，
避免 torchvision ToPILImage 对 float 的缩放回绕）。

统一协议差异（相对论文）：
- 200 个预先固定的基因（不是 2,000 HVG），PCA-50 只用训练切片拟合；
- 划分 train=A1+B1 / val=C1 / test=D1；
- batch_size=16（8 GB 显存适配），其余超参与官方一致
  （100 epochs, lr 5e-4, Fourier 128, sigma 1, dropout 0.3, patience 10）；
- 评测同时做推理 batch 大小稳定性审计（官方 SA 的注意力作用在 batch 上）。
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


DATA = ROOT / "data/processed/ressat_gse240429"
RESULTS = ROOT / "results/ressat_unified"
ARRAYS = ROOT / "data/processed/gse240429/arrays"
EXPERIMENT = "unified_seed42"
BATCH_SIZES = (1, 8, 16, 32, 64)


def clean_json(value):
    if isinstance(value, dict):
        return {key: clean_json(item) for key, item in value.items()}
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main() -> None:
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    train, validation, test = load_sections(str(DATA), section_num=4)
    model = ResSAT(
        train_sections=train,
        val_sections=validation,
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
    model.fit(
        num_epochs=100,
        lr=5e-4,
        batch_size=16,
        patience=10,
        weight_decay=1e-6,
    )

    # 评测：不同推理 batch 大小 + 反投影到 200 基因。
    evaluator = ResSAT(
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
    evaluator.load_checkpoint()
    with open(DATA / "pca_info.pkl", "rb") as source:
        pca_info = pickle.load(source)
    with np.load(ARRAYS / "C73_D1.npz") as source:
        coordinates = source["coordinates_xy"].astype(np.float32)
        genes = source["genes"]
        true_log_normalized = source["log_normalized"].astype(np.float32)

    predictions = {}
    metrics = {}
    for batch_size in BATCH_SIZES:
        predicted_pca, true_pca = evaluator.predict(batch_size=batch_size)
        predicted = back_project(predicted_pca, pca_info).numpy()
        true = back_project(true_pca, pca_info).numpy()
        predictions[batch_size] = predicted
        result, _ = evaluate_normalized(true, predicted, coordinates)
        metrics[str(batch_size)] = result
        print(f"batch={batch_size}: PCC={result['pcc_macro_all_fixed_genes']:.4f} "
              f"Spearman={result['spearman_macro_all_fixed_genes']:.4f}")

    reference = predictions[16]
    stability = {}
    for batch_size, predicted in predictions.items():
        difference = predicted - reference
        stability[str(batch_size)] = {
            "mean_absolute_difference_vs_batch16": float(np.abs(difference).mean()),
            "root_mean_squared_difference_vs_batch16": float(np.sqrt(np.mean(difference**2))),
            "maximum_absolute_difference_vs_batch16": float(np.abs(difference).max()),
            "flattened_prediction_correlation_vs_batch16": float(
                np.corrcoef(predicted.ravel(), reference.ravel())[0, 1]
            ),
        }

    np.savez_compressed(
        RESULTS / "unified_batch16_predictions.npz",
        predicted=reference.astype(np.float32),
        true=true_log_normalized,
        coordinates_xy=coordinates,
        genes=genes,
    )
    output = {
        "protocol": {
            "source": "GSE240429 unified adaptation",
            "train_sections": ["Section_3=C73_A1", "Section_4=C73_B1"],
            "validation_section": "Section_2=C73_C1",
            "test_section": "Section_1=C73_D1",
            "genes": 200,
            "target": "PCA-50 of unified log-normalized panel, fit on train only",
            "patch_format": "uint8 HWC stored by scripts/11 (torchvision-safe)",
            "training": "100 epochs, lr 5e-4, batch 16, Fourier 128, sigma 1, dropout 0.3",
            "default_inference_batch_size": 16,
        },
        "metrics_by_inference_batch_size": metrics,
        "batch_size_stability": stability,
    }
    (RESULTS / "evaluation.json").write_text(
        json.dumps(clean_json(output), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"完成：{RESULTS / 'evaluation.json'}")


if __name__ == "__main__":
    main()
