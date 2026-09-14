"""在补齐的 ResSAT 原始 SP 数据上运行一次作者协议复现。

Section_2 训练、Section_1 测试；100 epochs、batch 32 和模型超参数均采用
作者默认值。报告同时给出论文的 PCA/Harmony 重建口径和更严格的原始
log-normalized HVG 真值口径。
"""

from __future__ import annotations

import json
import os
import pickle
import random
import sys
import time
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


DATA = ROOT / "data/processed/ressat_original_rebuilt/SP"
RESULTS = ROOT / "results/ressat_sp_rebuilt"
EXPERIMENT = "sp_section2_to_section1_seed42"
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
    # 复用已下载的 ResNet50 权重，避免每个结果目录重复保存 98 MB。
    os.environ["TORCH_HOME"] = str(ROOT / "results/ressat_official/.cache/torch")
    train_sections, _, test_sections = load_sections(str(DATA), section_num=2)
    genes = load_gene_names(str(DATA))
    started = time.perf_counter()

    if TRAIN:
        trainer = ResSAT(
            train_sections=train_sections,
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
        trainer.fit(num_epochs=100, lr=5e-4, batch_size=32, patience=10, weight_decay=1e-6)
    training_seconds = time.perf_counter() - started

    evaluator = ResSAT(
        test_sections=test_sections,
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
    predicted_pca, true_pca = evaluator.predict(batch_size=32)
    with (DATA / "pca_info.pkl").open("rb") as handle:
        pca_info = pickle.load(handle)
    predicted = back_project(predicted_pca, pca_info).numpy().astype(np.float32)
    reconstructed_true = back_project(true_pca, pca_info).numpy().astype(np.float32)
    with np.load(DATA / "Section_1/strict_truth.npz") as source:
        strict_true = source["log_normalized"].astype(np.float32)
        coordinates = source["coordinates_xy"].astype(np.float32)
        gene_names = source["genes"]
    paper_metrics, _ = evaluate_normalized(reconstructed_true, predicted, coordinates)
    strict_metrics, _ = evaluate_normalized(strict_true, predicted, coordinates)
    highly_expressed = np.argsort(reconstructed_true.mean(axis=0))[-50:]
    correlations = [
        np.corrcoef(reconstructed_true[:, index], predicted[:, index])[0, 1]
        for index in highly_expressed
    ]
    paper_metrics["top50_high_expression_pcc"] = float(np.nanmean(correlations))
    RESULTS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        RESULTS / "SP1_predictions.npz",
        predicted=predicted,
        reconstructed_true=reconstructed_true,
        strict_true=strict_true,
        coordinates_xy=coordinates,
        genes=gene_names,
    )
    result = {
        "protocol": {
            "source": "10x raw SP sections, Methods-based preprocessing reconstruction",
            "train": "SP2 / Section_2",
            "test": "SP1 / Section_1",
            "seed": 42,
            "epochs": 100,
            "batch_size": 32,
            "genes": 2000,
            "runtime_seconds_including_training": training_seconds,
            "paper_reference": {"pcc_2000_hvg": 0.6980, "pcc_top50_heg": 0.8999},
        },
        "paper_reconstructed_space_metrics": paper_metrics,
        "strict_log_normalized_truth_metrics": strict_metrics,
    }
    (RESULTS / "evaluation.json").write_text(json.dumps(clean(result), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(clean(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
