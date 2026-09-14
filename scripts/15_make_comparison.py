"""汇总所有方法的统一协议结果：机器可读 JSON + Markdown 表 + 对比图。

只汇总"统一协议"（同一 GSE240429 划分、同一 200 基因面板、同一图像输入/评测）
的结果；ResSAT 官方示例的作者协议结果单独列出，不进统一排行榜。

读取的结果文件：
- results/nonimage_baselines/metrics.json
- results/unified_image_baselines/metrics.json
- results/ressat_unified/evaluation.json（batch=16 口径）
- results/genar_gse240429/C73_D1_inference_summary.json + predictions.npz
- results/stem_adapted/sampling_evaluation.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from he2st.metrics import evaluate_normalized


RESULTS = ROOT / "results"
ARRAYS = ROOT / "data/processed/gse240429/arrays"
OUTPUT_JSON = RESULTS / "unified_comparison.json"
OUTPUT_MD = RESULTS / "unified_comparison.md"
OUTPUT_FIG = ROOT / "figures/04_method_comparison.png"

UNIFIED_KEYS = [
    "pcc_macro_all_fixed_genes",
    "spearman_macro_all_fixed_genes",
    "mae_log_normalized",
    "rmse_log_normalized",
    "mean_spot_cosine",
    "relative_variation_distance",
    "moran_i_mae",
    "moran_i_gene_correlation",
]


def json_ready(value):
    if isinstance(value, dict):
        return {k: json_ready(v) for k, v in value.items()}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def load_metrics(path: Path, key: str = "test_metrics") -> dict:
    with open(path, encoding="utf-8") as source:
        data = json.load(source)
    return data.get(key, data)


def pick(method_metrics: dict) -> dict:
    return {key: method_metrics.get(key) for key in UNIFIED_KEYS}


def genar_unified_metrics() -> tuple[dict, dict]:
    """把 GenAR 的原始计数预测换算成统一 log 归一化空间后评测。"""

    npz = np.load(RESULTS / "genar_gse240429/C73_D1_predictions.npz")
    predicted_counts = npz["predicted_counts"].astype(np.float64)
    true_counts = npz["true_counts"].astype(np.float64)
    gene_names = npz["gene_names"]

    def log_normalize(counts: np.ndarray) -> np.ndarray:
        library = counts.sum(axis=1, keepdims=True)
        library[library <= 0] = 1.0
        return np.log1p(counts / library * 1e4).astype(np.float32)

    with np.load(ARRAYS / "C73_D1.npz") as source:
        coordinates = source["coordinates_xy"].astype(np.float32)
    metrics, per_gene_pcc = evaluate_normalized(
        log_normalize(true_counts), log_normalize(predicted_counts), coordinates
    )
    # 官方口径的 top-k（测试集内按 PCC 排序选择，偏乐观，单列不进入主表）。
    with open(
        RESULTS / "genar_gse240429/C73_D1_inference_summary.json", encoding="utf-8"
    ) as source:
        genar_summary = json.load(source)
    paper_metrics = {
        "paper_pcc_top10_test_selected": genar_summary["standard_metrics"].get("pcc_10"),
        "paper_pcc_top50_test_selected": genar_summary["standard_metrics"].get("pcc_50"),
        "paper_pcc_top200_test_selected": genar_summary["standard_metrics"].get("pcc_200"),
        "genar_average_final_scale_loss": genar_summary.get("average_final_scale_loss"),
        "genar_raw_count_mae": genar_summary["raw_count_diagnostics"].get("mae_count"),
        "genar_raw_count_zero_f1": genar_summary["raw_count_diagnostics"].get("zero_f1"),
    }
    return metrics, paper_metrics, per_gene_pcc


def main() -> None:
    table: dict[str, dict] = {}
    extra: dict[str, dict] = {}

    # 1) 非图像基线
    nonimage = load_metrics(RESULTS / "nonimage_baselines/metrics.json")
    for name in ("train_gene_mean", "coordinate_ridge", "coordinate_knn"):
        table[name] = pick(nonimage[name])

    # 2) 图像基线
    image = load_metrics(RESULTS / "unified_image_baselines/metrics.json")
    for name in ("image_ridge", "stnet_style_mlp", "bleep"):
        table[name] = pick(image[name])

    # 3) ResSAT 统一适配：主口径用统一 npz（预测 vs 完整真值），
    #    PCA 子空间口径（官方评测方式，真值也经 PCA-50 重建）单列为参考。
    if (RESULTS / "ressat_unified/evaluation.json").exists():
        ressat = json.loads(
            (RESULTS / "ressat_unified/evaluation.json").read_text(encoding="utf-8")
        )
        table["ressat_unified"] = pick(ressat["metrics_by_inference_batch_size"]["16"])
        extra["ressat_unified"] = {
            "pca_subspace_pcc_batch16": ressat["metrics_by_inference_batch_size"]["16"][
                "pcc_macro_all_fixed_genes"
            ],
            "note": "evaluation.json 的口径是预测与'真值'都经 PCA-50 重建（官方口径）；"
                    "统一表中的 PCC 是对完整 200 基因真值重算的严格口径。",
        }
    if (RESULTS / "ressat_unified/unified_batch16_predictions.npz").exists():
        ressat_npz = np.load(RESULTS / "ressat_unified/unified_batch16_predictions.npz")
        with np.load(ARRAYS / "C73_D1.npz") as source:
            coordinates = source["coordinates_xy"].astype(np.float32)
        ressat_full, _ = evaluate_normalized(
            ressat_npz["true"].astype(np.float32),
            ressat_npz["predicted"].astype(np.float32),
            coordinates,
        )
        table["ressat_unified"] = pick(ressat_full)

    # 4) GenAR 统一适配
    if (RESULTS / "genar_gse240429/C73_D1_predictions.npz").exists():
        genar_metrics, genar_extra, _ = genar_unified_metrics()
        table["genar"] = pick(genar_metrics)
        extra["genar"] = genar_extra

    # 5) Stem 统一适配
    if (RESULTS / "stem_adapted/sampling_evaluation.json").exists():
        stem = json.loads(
            (RESULTS / "stem_adapted/sampling_evaluation.json").read_text(encoding="utf-8")
        )
        table["stem"] = pick(stem["test_metrics"])
        extra["stem"] = {
            "per_gene_sample_vs_true_rvd": stem.get("per_gene_sample_vs_true_rvd"),
            "samples_per_spot": stem["protocol"]["samples_per_spot"],
            "sampling_steps": stem["protocol"]["sampling_steps"],
        }

    # 6) ResSAT 官方示例（作者协议，单列）
    author_protocol = None
    if (RESULTS / "ressat_official/evaluation.json").exists():
        ressat_official = json.loads(
            (RESULTS / "ressat_official/evaluation.json").read_text(encoding="utf-8")
        )
        author_protocol = {
            "source": "ResSAT official Zenodo example (mouse brain), author protocol",
            "default_batch": 32,
            "pcc_macro_all_genes": ressat_official["metrics_by_inference_batch_size"]["32"][
                "pcc_macro_all_fixed_genes"
            ],
            "official_top50_high_expression_pcc": ressat_official["metrics_by_inference_batch_size"]["32"][
                "official_top50_high_expression_pcc"
            ],
            "batch_size_stability": ressat_official["batch_size_stability"],
        }

    output = {
        "protocol": {
            "dataset": "GSE240429 human liver 10x Visium",
            "train": ["C73_A1", "C73_B1"],
            "validation": "C73_C1",
            "test": "C73_D1",
            "genes": "200 fixed panel (SHA256 in data/processed/gse240429/manifest.json)",
            "image_input": "224x224 patches around each spot; shared ResNet18 features where applicable",
            "metrics": UNIFIED_KEYS,
            "note": "ResSAT/Stem/GenAR entries are unified-protocol adaptations "
                    "(same data/split/genes), not the paper's original numbers.",
        },
        "unified_table": table,
        "extra_metrics": extra,
        "author_protocol_reference": author_protocol,
    }
    OUTPUT_JSON.write_text(
        json.dumps(json_ready(output), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # Markdown 表
    lines = [
        "# 统一协议方法比较（GSE240429, test=C73_D1）",
        "",
        "所有方法：同一训练/验证/测试切片、同一 200 基因面板、同一评测代码。",
        "ResSAT/Stem/GenAR 为统一数据上的适配实现，不是论文原数值。",
        "",
        "| 方法 | PCC↑ | Spearman↑ | MAE↓ | RMSE↓ | spot cos↑ | RVD↓ | Moran MAE↓ | Moran corr↑ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    order = [
        "train_gene_mean", "coordinate_ridge", "coordinate_knn",
        "image_ridge", "stnet_style_mlp", "bleep",
        "ressat_unified", "genar", "stem",
    ]

    def fmt(value: float | None) -> str:
        return f"{value:.4f}" if isinstance(value, float) else "—"

    for name in order:
        if name not in table:
            continue
        row = table[name]
        lines.append(
            f"| {name} | {fmt(row['pcc_macro_all_fixed_genes'])} "
            f"| {fmt(row['spearman_macro_all_fixed_genes'])} "
            f"| {fmt(row['mae_log_normalized'])} | {fmt(row['rmse_log_normalized'])} "
            f"| {fmt(row['mean_spot_cosine'])} | {fmt(row['relative_variation_distance'])} "
            f"| {fmt(row['moran_i_mae'])} | {fmt(row['moran_i_gene_correlation'])} |"
        )
    lines += [
        "",
        "注：PCC/Spearman 为 200 基因逐基因相关性的宏平均；RVD 越低越好；",
        "Moran MAE 为逐基因空间自相关绝对误差，Moran corr 为真值-预测 Moran's I 的跨基因相关。",
    ]
    if author_protocol is not None:
        lines += [
            "",
            "## 作者协议参考（不进入统一排行榜）",
            "",
            f"- ResSAT 官方示例（鼠脑 2 切片，2,000 基因，官方预处理）：",
            f"  全基因宏平均 PCC={author_protocol['pcc_macro_all_genes']:.4f}，",
            f"  官方 top-50 高表达基因 PCC={author_protocol['official_top50_high_expression_pcc']:.4f}；",
            f"  推理 batch 1/8/16/32/64 的 PCC 相差 ≤ {max(abs(v['flattened_prediction_correlation_vs_batch32'] - 1) for v in author_protocol['batch_size_stability'].values()):.5f}。",
        ]
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # 对比图：PCC 柱状图 + Spearman（PCC 无定义的方法除外）
    available = [
        name
        for name in order
        if name in table and isinstance(table[name]["pcc_macro_all_fixed_genes"], float)
    ]
    pcc = [table[name]["pcc_macro_all_fixed_genes"] for name in available]
    spearman = [table[name]["spearman_macro_all_fixed_genes"] for name in available]
    labels = [name.replace("_", "\n") for name in available]
    x = np.arange(len(available))
    width = 0.38
    figure, axis = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
    axis.bar(x - width / 2, pcc, width, label="PCC macro (200 genes)", color="#3d5a80")
    axis.bar(x + width / 2, spearman, width, label="Spearman macro", color="#98c1d9")
    axis.axhline(0, color="#888", linewidth=0.8)
    axis.set_xticks(x)
    axis.set_xticklabels(labels, fontsize=9)
    axis.set_ylabel("correlation")
    axis.set_title("Unified-protocol comparison on GSE240429 (test slide C73_D1)")
    axis.legend()
    axis.grid(axis="y", alpha=0.3)
    figure.savefig(OUTPUT_FIG, dpi=180)
    plt.close(figure)

    print(f"完成：{OUTPUT_JSON}")
    print(f"完成：{OUTPUT_MD}")
    print(f"完成：{OUTPUT_FIG}")


if __name__ == "__main__":
    main()
