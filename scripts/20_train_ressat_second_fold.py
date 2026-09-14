"""补跑 ResSAT 的第二个严格留出切片：train=A1+B1, val=D1, test=C1。

数据、PCA 与 scripts/12 完全相同，只交换 C1/D1 的验证和测试角色。脚本无参数；
已有 checkpoint 时可将 TRAIN 改为 False，仅重新评测。
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
ARRAYS = ROOT / "data/processed/gse240429/arrays"
RESULTS = ROOT / "results/ressat_second_fold"
EXPERIMENT = "test_C1_seed42"
TRAIN = True


def clean(value):
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, np.generic):
        value = value.item()
    return None if isinstance(value, float) and not np.isfinite(value) else value


def main() -> None:
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    train_sections, c1_sections, d1_sections = load_sections(str(DATA), section_num=4)
    genes = load_gene_names(str(DATA))

    if TRAIN:
        trainer = ResSAT(
            train_sections=train_sections,
            val_sections=d1_sections,
            data_dir=str(DATA),
            result_dir=str(RESULTS),
            patch_size=224,
            num_fourier=128,
            sigma=1,
            dropout=0.3,
            num_workers=4,
            exp_name=EXPERIMENT,
            gene_names=genes,
        )
        trainer.fit(num_epochs=100, lr=5e-4, batch_size=16, patience=10, weight_decay=1e-6)

    evaluator = ResSAT(
        test_sections=c1_sections,
        data_dir=str(DATA),
        result_dir=str(RESULTS),
        patch_size=224,
        num_fourier=128,
        sigma=1,
        dropout=0.3,
        num_workers=4,
        exp_name=EXPERIMENT,
        gene_names=genes,
    )
    evaluator.load_checkpoint()
    predicted_pca, true_pca = evaluator.predict(batch_size=16)
    with (DATA / "pca_info.pkl").open("rb") as source:
        pca_info = pickle.load(source)
    predicted = back_project(predicted_pca, pca_info).numpy().astype(np.float32)
    pca_true = back_project(true_pca, pca_info).numpy().astype(np.float32)
    with np.load(ARRAYS / "C73_C1.npz") as source:
        true = source["log_normalized"].astype(np.float32)
        coordinates = source["coordinates_xy"].astype(np.float32)
        gene_names = source["genes"]
    strict_metrics, _ = evaluate_normalized(true, predicted, coordinates)
    pca_metrics, _ = evaluate_normalized(pca_true, predicted, coordinates)
    RESULTS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        RESULTS / "C73_C1_predictions.npz",
        predicted=predicted,
        true=true,
        pca_reconstructed_true=pca_true,
        coordinates_xy=coordinates,
        genes=gene_names,
    )
    report = {
        "protocol": {
            "train": ["C73_A1", "C73_B1"],
            "validation": "C73_D1",
            "test": "C73_C1",
            "genes": 200,
            "pca_components": 50,
            "inference_batch_size": 16,
        },
        "strict_full_expression_metrics": strict_metrics,
        "pca_reconstructed_truth_metrics": pca_metrics,
    }
    (RESULTS / "evaluation.json").write_text(json.dumps(clean(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(clean(report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
