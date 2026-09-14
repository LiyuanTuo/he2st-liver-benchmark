"""HEG 基准最终评估：旧面板 vs HEG 面板（原空间 + PCA-50 重建空间 + top-50 HEG）。

复用已有预测文件，不训练。输出：
- results/heg_benchmark_final.json
- figures/benchmark_ppt/heg_final_comparison.png
- figures/benchmark_ppt/cause_decomposition_v2.png

运行：py -3.12 scripts/44_evaluate_heg_benchmark.py
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
RESULTS = ROOT / "results"
ASSETS = ROOT / "figures/benchmark_ppt"
OUT_JSON = RESULTS / "heg_benchmark_final.json"

plt.rcParams.update({"font.family": "Microsoft YaHei", "axes.unicode_minus": False})
RED, BLUE, GOLD, INK, GRAY = "#B71C1C", "#6C8EBF", "#B08D57", "#262626", "#9A9A9A"

OLD_PANEL = "data/processed/gse240429/arrays"
HEG_PANEL = "data/processed/gse240429_heg/arrays"

METHODS = [
    ("Image Ridge", "image_ridge", "log", "unified_image_baselines_heg/image_ridge_C73_D1.npz", "predicted"),
    ("MLP 基线", "stnet_style_mlp", "log", "unified_image_baselines_heg/stnet_style_mlp_C73_D1.npz", "predicted"),
    ("BLEEP", "bleep", "log", "unified_image_baselines_heg/bleep_C73_D1.npz", "predicted"),
    ("ResSAT", "ressat", "log", "ressat_unified_heg/unified_batch16_predictions.npz", "predicted"),
    ("GenAR(改)", "genar", "count", "improvement_heg/genar/C73_D1_predictions.npz", "predicted_counts"),
    ("Stem(改)", "stem", "log", "improvement_heg/stem/C73_D1_predictions.npz", "predicted"),
]
FALLBACK = {
    "GenAR(改)": ("genar_gse240429_heg/C73_D1_predictions.npz", "predicted_counts", "count"),
    "Stem(改)": ("stem_adapted_heg/stem_samples_D1.npz", "mean_prediction", "log"),
}


def panel_normalize(values):
    values = np.maximum(np.asarray(values, dtype=np.float64), 0)
    return np.log1p(10000 * values / np.maximum(values.sum(1, keepdims=True), 1e-12))


def gene_pcc(truth, prediction):
    x, y = truth - truth.mean(0), prediction - prediction.mean(0)
    den = np.sqrt((x * x).sum(0) * (y * y).sum(0))
    valid = (np.ptp(truth, axis=0) > 1e-10) & (np.ptp(prediction, axis=0) > 1e-10)
    return np.divide((x * y).sum(0), den, out=np.full(truth.shape[1], np.nan), where=valid)


def fit_pca50(train_matrix):
    mean = train_matrix.mean(axis=0)
    std = train_matrix.std(axis=0)
    std[std < 1e-8] = 1.0
    standardized = (train_matrix - mean) / std
    u, s, vt = np.linalg.svd(standardized, full_matrices=False)
    return mean, std, vt[:50]


def reconstruct(matrix, mean, std, components):
    scores = (matrix - mean) / std @ components.T
    return mean + std * (scores @ components)


def load_truth(panel):
    with np.load(ROOT / panel / "C73_D1.npz") as data:
        return data["raw_counts"], data["genes"]


def load_prediction(relative, key, kind):
    with np.load(RESULTS / relative) as data:
        values = data[key].astype(np.float64)
    return panel_normalize(values if kind == "count" else np.expm1(values))


def main():
    # ---- HEG 面板真值与预测 ----
    raw_heg, genes_heg = load_truth(HEG_PANEL)
    truth = panel_normalize(raw_heg)
    predictions, sources = {}, {}
    for label, _, kind, relative, key in METHODS:
        path = RESULTS / relative
        if not path.exists() and label in FALLBACK:
            relative, key, kind = FALLBACK[label]
            path = RESULTS / relative
        if not path.exists():
            predictions[label] = None
            sources[label] = "MISSING"
            continue
        predictions[label] = load_prediction(relative, key, kind)
        sources[label] = relative
    with np.load(ROOT / HEG_PANEL / "C73_D1.npz") as data:
        genes_heg = data["genes"]

    # ---- PCA-50（HEG 训练面板拟合）----
    train_heg = np.concatenate(
        [panel_normalize(np.load(ROOT / HEG_PANEL / f"C73_{s}.npz")["raw_counts"])
         for s in ("A1", "B1")], axis=0)
    mean, std, components = fit_pca50(train_heg)
    truth_recon = reconstruct(truth, mean, std, components)
    recon = {name: reconstruct(pred, mean, std, components)
             for name, pred in predictions.items() if pred is not None}

    # ---- 基因子集（面板内 top-50 HEG = 全转录组 top-50，面板即按表达排序选出）----
    # Use the same whole-transcriptome-normalized scale as panel construction.
    train_selection_values = np.concatenate([
        np.load(ROOT / HEG_PANEL / f"C73_{s}.npz")["log_normalized"] for s in ("A1", "B1")])
    order_train = np.argsort(train_selection_values.mean(axis=0))
    order_test = np.argsort(truth.mean(axis=0))
    subsets = {
        "all_200": np.arange(200),
        "top50_heg_train_selected": order_train[-50:],
        "top50_heg_test_selected": order_test[-50:],
    }

    table = {"raw": {}, "pca50_reconstructed": {}}
    per_gene = {}
    for space, (space_truth, space_pred) in {
        "raw": (truth, predictions),
        "pca50_reconstructed": (truth_recon, recon),
    }.items():
        for name, pred in space_pred.items():
            if pred is None:
                table[space][name] = {k: None for k in subsets}
                continue
            pcc = gene_pcc(space_truth, pred)
            if space == "raw" and name == "ResSAT":
                per_gene["ResSAT"] = {"pcc": pcc, "expression": truth.mean(axis=0)}
            table[space][name] = {
                key: float(np.nanmean(pcc[idx])) for key, idx in subsets.items()
            }
            table[space][name]["mae"] = float(np.abs(space_truth - pred).mean())

    # ---- 旧面板（原空间 all-200）供对比 ----
    old_audit = json.loads((RESULTS / "benchmark_ppt_audit.json").read_text(encoding="utf-8"))
    OLD_LABELS = {"GenAR(改)": "GenAR", "Stem(改)": "Stem"}
    old_raw = {label: old_audit["stats"][OLD_LABELS.get(label, label)]["pcc"]
               for label, *_ in METHODS}

    output = {
        "protocol": {
            "old_panel": "200 variance-selected genes, raw log space (2026-09-08 deck)",
            "heg_panel": "top-200 HEGs by train mean expression, all methods retrained",
            "spaces": ["raw panel-normalized", "PCA-50 reconstructed (fit on train)"],
            "subsets": ["all 200 genes", "top-50 HEG (train-selected, leak-free)",
                        "top-50 HEG (test-selected, paper style)"],
            "note": "GenAR/BLEEP/Stem entries use improved variants when available, "
                    "fallback to base retrain; see sources.",
        },
        "sources": sources,
        "old_panel_raw_all200": old_raw,
        "heg_panel": table,
        "genes": genes_heg.tolist(),
    }
    (ROOT / OUT_JSON).write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))

    missing = [label for label, *_ in METHODS if table["raw"][label]["all_200"] is None]
    if missing:
        raise ValueError(f"以下方法缺少 HEG 预测，请先完成重训/推理：{missing}")

    # ---- 图 1：旧 vs HEG 面板对比 ----
    names = [label for label, *_ in METHODS]
    values_old = [old_raw.get(n) for n in names]
    values_heg_raw = [table["raw"][n]["all_200"] for n in names]
    values_heg_top50 = [table["pca50_reconstructed"][n]["top50_heg_train_selected"] for n in names]
    x = np.arange(len(names))
    width = 0.26
    fig, ax = plt.subplots(figsize=(12.4, 4.4))
    ax.bar(x - width, values_old, width, color=GRAY, label="旧面板 · 原空间 · 全 200 基因")
    ax.bar(x, values_heg_raw, width, color=BLUE, label="HEG 面板 · 原空间 · 全 200 基因")
    ax.bar(x + width, values_heg_top50, width, color=RED,
           label="HEG 面板 · PCA-50 重建空间 · top-50 HEG")
    for i, name in enumerate(names):
        for offset, source in [(-width, values_old), (0, values_heg_raw), (width, values_heg_top50)]:
            value = source[i]
            if value is None:
                continue
            ax.text(i + offset, value + 0.012, f"{value:.3f}", ha="center", fontsize=11)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=14)
    ax.set_ylabel("平均 PCC", fontsize=13)
    ax.set_ylim(0, max(v for v in values_heg_top50 if v is not None) * 1.26)
    ax.legend(fontsize=12.5, frameon=False, ncol=3)
    ax.grid(axis="y", alpha=.25)
    ax.set_title("同一测试切片 C73_D1 · 旧面板与高表达基因（HEG）面板重训结果对比",
                 fontsize=15, color=INK)
    fig.savefig(ASSETS / "heg_final_comparison.png", dpi=185, facecolor="white")
    plt.close(fig)

    # ---- 图 2：HEG 面板内表达水平 vs 逐基因 PCC ----
    if per_gene:
        pcc = per_gene["ResSAT"]["pcc"]
        expression = per_gene["ResSAT"]["expression"]
        fig, ax = plt.subplots(figsize=(5.6, 3.9))
        ax.scatter(expression, pcc, s=34, color=RED, rasterized=True)
        ax.axhline(np.nanmean(pcc), color=GOLD, ls="--", lw=1.4,
                   label=f"平均 PCC = {np.nanmean(pcc):.3f}")
        ax.set_xlabel("平均表达（面板内 log1p）", fontsize=12)
        ax.set_ylabel("逐基因 PCC（原始空间）", fontsize=12)
        ax.legend(fontsize=11, frameon=False)
        ax.set_title("HEG 面板（200 个高表达基因）", fontsize=13, color=INK)
        fig.savefig(ASSETS / "heg_panel_scatter.png", dpi=185, facecolor="white")
        plt.close(fig)

    make_cause_v2()
    print("figures saved")


def make_cause_v2():
    """原因分解图 v2：旧面板散点 + HEG 面板散点 + 三口径阶梯。"""
    # 旧面板 ResSAT（原空间 + 重建空间）
    with np.load(ROOT / OLD_PANEL / "C73_D1.npz") as data:
        raw_old, genes_old = data["raw_counts"], data["genes"]
    truth_old = panel_normalize(raw_old)
    with np.load(RESULTS / "ressat_unified/unified_batch16_predictions.npz") as data:
        pred_old_log = data["predicted"].astype(np.float64)
    pred_old = panel_normalize(np.expm1(pred_old_log))
    pcc_old_raw = gene_pcc(truth_old, pred_old)
    train_old = np.concatenate(
        [panel_normalize(np.load(ROOT / OLD_PANEL / f"C73_{s}.npz")["raw_counts"])
         for s in ("A1", "B1")], axis=0)
    mean_old, std_old, comp_old = fit_pca50(train_old)
    truth_old_recon = reconstruct(truth_old, mean_old, std_old, comp_old)
    pred_old_recon = reconstruct(pred_old, mean_old, std_old, comp_old)
    pcc_old_recon = gene_pcc(truth_old_recon, pred_old_recon)

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.4))
    fig.subplots_adjust(left=.055, right=.975, top=.88, bottom=.16, wspace=.28)

    ax = axes[0]
    expression_old = truth_old.mean(axis=0)
    threshold = np.sort(expression_old)[-50]
    top50 = expression_old >= threshold
    ax.scatter(expression_old[~top50], pcc_old_raw[~top50], s=24, color="#C9C9C9",
               label="其余 150 个基因", rasterized=True)
    ax.scatter(expression_old[top50], pcc_old_raw[top50], s=34, color=RED,
               label="面板内 top-50 表达", rasterized=True)
    ax.axvline(threshold, color=GOLD, ls="--", lw=1.4)
    ax.set_xlabel("平均表达（log1p）", fontsize=12)
    ax.set_ylabel("逐基因 PCC（原空间）", fontsize=12)
    ax.set_title(f"① 旧面板（按方差选）：低表达基因噪声主导，\n全 200 平均 = {np.nanmean(pcc_old_raw):.3f}",
                 fontsize=13.5, color=INK)
    ax.legend(fontsize=10, frameon=False)
    ax.tick_params(labelsize=10)

    ax = axes[1]
    pcc_heg = gene_pcc(truth, predictions["ResSAT"])
    ax.scatter(truth.mean(axis=0), pcc_heg, s=34, color=BLUE, rasterized=True)
    ax.set_xlabel("平均表达（log1p）", fontsize=12)
    ax.set_ylabel("逐基因 PCC（原空间）", fontsize=12)
    ax.set_title(f"② HEG 面板（按表达选）：整体上移，\n全 200 平均 = {np.nanmean(pcc_heg):.3f}",
                 fontsize=13.5, color=INK)
    ax.tick_params(labelsize=10)

    ax = axes[2]
    labels = ["旧面板\n原空间", "旧面板\nPCA-50 重建", "HEG 面板重建空间\ntop-50 HEG"]
    values = [np.nanmean(pcc_old_raw), np.nanmean(pcc_old_recon),
              table["pca50_reconstructed"]["ResSAT"]["top50_heg_train_selected"]]
    colors = [GRAY, BLUE, RED]
    bars = ax.bar(labels, values, color=colors, width=.58)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + .012, f"{value:.3f}",
                ha="center", fontsize=13.5, weight="bold")
    ax.set_ylim(0, max(values) * 1.24)
    ax.set_ylabel("ResSAT 平均 PCC", fontsize=12)
    ax.set_title("③ 同一模型实现，只换面板与口径", fontsize=13.5, color=INK)
    ax.tick_params(labelsize=10.5)
    fig.savefig(ASSETS / "cause_decomposition_v2.png", dpi=185, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
