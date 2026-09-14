"""对两张人肝测试切片上的主要方法做配对基因层面比较。

每个统计单位是“同一个基因在同一张测试切片上的 PCC”。C1 与 D1 各有
200 个固定基因，因此共有 400 个配对单位。Bootstrap 只用于描述当前两张
切片上的基因分布，不能替代更多患者上的置信区间。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

FILES = {
    "Image Ridge": {
        "C1": RESULTS / "cross_slide_robustness/image_ridge_C73_C1.npz",
        "D1": RESULTS / "cross_slide_robustness/image_ridge_C73_D1.npz",
    },
    "MLP ensemble": {
        "C1": RESULTS / "cross_slide_robustness/stnet_style_mlp_C73_C1_ensemble.npz",
        "D1": RESULTS / "cross_slide_robustness/stnet_style_mlp_C73_D1_ensemble.npz",
    },
    "BLEEP ensemble": {
        "C1": RESULTS / "cross_slide_robustness/bleep_C73_C1_ensemble.npz",
        "D1": RESULTS / "cross_slide_robustness/bleep_C73_D1_ensemble.npz",
    },
    "ResSAT": {
        "C1": RESULTS / "ressat_second_fold/C73_C1_predictions.npz",
        "D1": RESULTS / "ressat_unified/unified_batch16_predictions.npz",
    },
}


def per_gene_pcc(true: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    true = true - true.mean(axis=0, keepdims=True)
    predicted = predicted - predicted.mean(axis=0, keepdims=True)
    numerator = (true * predicted).sum(axis=0)
    denominator = np.sqrt((true * true).sum(axis=0) * (predicted * predicted).sum(axis=0))
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)


def load_values(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as source:
        true = source["true"].astype(np.float64)
        predicted = source["predicted"].astype(np.float64)
        genes = source["genes"].astype(str)
    return genes, per_gene_pcc(true, predicted)


def percentile_interval(samples: np.ndarray) -> list[float]:
    return [float(x) for x in np.percentile(samples, [2.5, 97.5])]


def main() -> None:
    rng = np.random.default_rng(20260907)
    values: dict[str, np.ndarray] = {}
    reference_genes = None
    for method, slides in FILES.items():
        per_slide = []
        for slide in ("C1", "D1"):
            genes, pcc = load_values(slides[slide])
            if reference_genes is None:
                reference_genes = genes
            elif not np.array_equal(reference_genes, genes):
                raise ValueError(f"基因顺序不一致：{method}/{slide}")
            per_slide.append(pcc)
        values[method] = np.concatenate(per_slide)

    matrix = np.stack(list(values.values()))
    if not np.isfinite(matrix).all():
        raise ValueError("PCC 中存在未定义值，不能执行配对 bootstrap")
    draws = rng.integers(0, matrix.shape[1], size=(20_000, matrix.shape[1]))
    boot_means = matrix[:, draws].mean(axis=2)

    summary = {
        "unit": "gene x held-out slide (200 genes x C1/D1 = 400 paired units)",
        "bootstrap_resamples": 20_000,
        "interpretation_limit": "descriptive gene-level interval; not a patient-level confidence interval",
        "methods": {},
        "paired_differences_vs_ressat": {},
    }
    names = list(values)
    for index, method in enumerate(names):
        arr = values[method]
        summary["methods"][method] = {
            "mean_pcc": float(arr.mean()),
            "median_pcc": float(np.median(arr)),
            "bootstrap_95ci_of_mean": percentile_interval(boot_means[index]),
            "positive_fraction": float((arr > 0).mean()),
        }
    ressat_index = names.index("ResSAT")
    for index, method in enumerate(names[:-1]):
        difference = values["ResSAT"] - values[method]
        boot_difference = boot_means[ressat_index] - boot_means[index]
        summary["paired_differences_vs_ressat"][method] = {
            "mean_difference_ressat_minus_method": float(difference.mean()),
            "bootstrap_95ci": percentile_interval(boot_difference),
            "ressat_higher_fraction": float((difference > 0).mean()),
            "bootstrap_samples_crossing_zero": int((boot_difference <= 0).sum()),
        }

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "statistical_comparison.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    colors = ["#3F5F89", "#91BCD2", "#B08D57", "#B71C1C"]
    fig, axis = plt.subplots(figsize=(10.5, 5.5))
    parts = axis.violinplot([values[name] for name in names], showextrema=False, showmedians=False)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.24)
    positions = np.arange(1, len(names) + 1)
    means = [summary["methods"][name]["mean_pcc"] for name in names]
    lows = [summary["methods"][name]["bootstrap_95ci_of_mean"][0] for name in names]
    highs = [summary["methods"][name]["bootstrap_95ci_of_mean"][1] for name in names]
    axis.errorbar(positions, means, yerr=[np.asarray(means) - lows, np.asarray(highs) - means],
                  fmt="o", color="#262626", capsize=5, linewidth=1.5, zorder=3)
    axis.axhline(0, color="#888888", linewidth=0.8)
    axis.set_xticks(positions, names)
    axis.set_ylabel("Per-gene PCC")
    axis.set_title("Human liver C1+D1: paired gene-level PCC distributions")
    axis.grid(axis="y", alpha=0.2)
    for position, mean in zip(positions, means):
        axis.text(position, mean + 0.035, f"mean={mean:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "09_gene_level_statistical_comparison.png", dpi=220)
    plt.close(fig)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
