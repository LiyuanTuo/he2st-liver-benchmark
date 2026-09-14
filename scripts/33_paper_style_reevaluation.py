"""按 ResSAT 论文口径重评统一基准：PCA-50 重建空间 + 高表达基因（HEG）子集。

原因量化 + 改进重算，全部复用已有预测，不训练：
1) 原始空间逐基因 PCC 与平均表达的关系（scripts/32 已算，此处补充重建空间）；
2) 全转录组 top-50 HEG（训练集选择，无泄漏）与 200 基因面板的重叠；
3) 用训练切片面板内归一化表达拟合 PCA-50，把真值与六法预测投影并重建，
   重算：全 200 基因 / 面板内 top-50 HEG / 全转录组 top-50 HEG∩面板。

输出：
- results/paper_style_reevaluation.json
- figures/benchmark_ppt/cause_decomposition.png   口径差异分解图
- figures/benchmark_ppt/heg_benchmark_bars.png    全方法 × 口径 对比柱状图

运行：py -3.12 scripts/33_paper_style_reevaluation.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import mmread
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
ARRAYS = ROOT / "data/processed/gse240429/arrays"
ASSETS = ROOT / "figures/benchmark_ppt"
OUT_JSON = RESULTS / "paper_style_reevaluation.json"
BLEEP_DATA = ROOT / "third_party/BLEEP/GSE240429_data/data"

plt.rcParams.update({"font.family": "Microsoft YaHei", "axes.unicode_minus": False})
RED, BLUE, GOLD, INK, GRAY = "#B71C1C", "#6C8EBF", "#B08D57", "#262626", "#8A8A8A"

SLIDE_NUM = {"C73_A1": "1", "C73_B1": "2", "C73_C1": "3", "C73_D1": "4"}
METHODS = [
    ("Image Ridge", "unified_image_baselines/image_ridge_C73_D1.npz", "predicted", False),
    ("MLP 基线", "unified_image_baselines/stnet_style_mlp_C73_D1.npz", "predicted", False),
    ("BLEEP", "unified_image_baselines/bleep_C73_D1.npz", "predicted", False),
    ("ResSAT", "ressat_unified/unified_batch16_predictions.npz", "predicted", False),
    ("GenAR", "genar_gse240429/C73_D1_predictions.npz", "predicted_counts", True),
    ("Stem", "stem_adapted/stem_samples_D1.npz", "mean_prediction", False),
]
IMPROVED = [
    ("BLEEP(改)", "improvement_20260908/bleep/C73_D1_predictions.npz", "predicted", False),
    ("GenAR(改)", "improvement_20260908/genar/C73_D1_predictions.npz", "predicted_counts", True),
    ("Stem(改)", "improvement_20260908/stem/C73_D1_predictions.npz", "predicted", False),
]


def panel_normalize(values):
    values = np.maximum(np.asarray(values, dtype=np.float64), 0)
    return np.log1p(10000 * values / np.maximum(values.sum(1, keepdims=True), 1e-12))


def gene_pcc(truth, prediction):
    x, y = truth - truth.mean(0), prediction - prediction.mean(0)
    den = np.sqrt((x * x).sum(0) * (y * y).sum(0))
    valid = (np.ptp(truth, axis=0) > 1e-10) & (np.ptp(prediction, axis=0) > 1e-10)
    return np.divide((x * y).sum(0), den, out=np.full(truth.shape[1], np.nan), where=valid)


def load_full_transcriptome(slide):
    """返回 (sparse counts, gene names, barcodes)。矩阵按 features.tsv 顺序。"""
    folder = BLEEP_DATA / f"filtered_expression_matrices/{SLIDE_NUM[slide]}"
    counts = mmread(folder / "matrix.mtx").tocsr().T.astype(np.int32)
    genes = np.asarray(
        [line.split("\t")[1] for line in (folder / "features.tsv").read_text(encoding="utf-8").splitlines()],
        dtype=object,
    )
    return counts, genes


def full_log_normalize(counts):
    library = np.asarray(counts.sum(axis=1)).ravel().astype(np.float64)
    library[library <= 0] = 1.0
    return np.log1p(counts.astype(np.float64).multiply((1e4 / library)[:, None])).tocsr()


def fit_pca50(train_matrix):
    mean = train_matrix.mean(axis=0)
    std = train_matrix.std(axis=0)
    std[std < 1e-8] = 1.0
    standardized = (train_matrix - mean) / std
    u, s, vt = np.linalg.svd(standardized, full_matrices=False)
    components = vt[:50]
    return mean, std, components


def reconstruct(matrix, mean, std, components):
    scores = (matrix - mean) / std @ components.T
    return mean + std * (scores @ components)


def load_predictions():
    raw_d1, genes = None, None
    with np.load(ARRAYS / "C73_D1.npz") as data:
        raw_d1, genes = data["raw_counts"], data["genes"]
    truth = panel_normalize(raw_d1)
    predictions = {}
    for name, relative, key, is_count in METHODS + IMPROVED:
        with np.load(RESULTS / relative) as data:
            values = data[key].astype(np.float64)
            npz_genes = data["gene_names"] if is_count else data["genes"]
            assert np.array_equal(npz_genes, genes), name
        prediction = panel_normalize(values if is_count else np.expm1(values))
        predictions[name] = prediction
    return raw_d1, genes, truth, predictions


def main():
    raw_d1, genes, truth, predictions = load_predictions()

    # ---- 1) 全转录组 top-50 HEG：训练集平均 log1p(1e4 归一化) 表达 ----
    train_counts = [load_full_transcriptome(s)[0] for s in ("C73_A1", "C73_B1")]
    features = load_full_transcriptome("C73_A1")[1]
    train_norm = sparse.vstack([full_log_normalize(c) for c in train_counts])
    full_mean = np.asarray(train_norm.mean(axis=0)).ravel()
    order = np.argsort(full_mean)[::-1]
    top50_heg = features[order[:50]].tolist()
    top200_heg = features[order[:200]].tolist()
    panel = genes.tolist()
    overlap50 = [g for g in top50_heg if g in panel]
    overlap200 = [g for g in top200_heg if g in panel]

    # ---- 2) PCA-50 重建空间（面板内归一化训练表达拟合，无测试泄漏）----
    train_panel = np.concatenate(
        [panel_normalize(np.load(ARRAYS / f"C73_{s}.npz")["raw_counts"]) for s in ("A1", "B1")],
        axis=0,
    )
    mean, std, components = fit_pca50(train_panel)
    truth_recon = reconstruct(truth, mean, std, components)
    recon = {
        name: reconstruct(prediction, mean, std, components)
        for name, prediction in predictions.items()
    }

    # ---- 3) 口径矩阵：空间 × 基因子集 ----
    def rank_indices(values):
        return np.argsort(values)  # 升序

    order_test = rank_indices(truth.mean(axis=0))
    order_train = rank_indices(train_panel.mean(axis=0))
    overlap_idx = np.asarray([panel.index(g) for g in overlap50], dtype=int) if overlap50 else np.zeros(0, dtype=int)

    spaces = {"raw": (truth, predictions), "pca50_reconstructed": (truth_recon, recon)}
    output = {"top50_heg_genomewide_train_selected": top50_heg,
              "overlap_with_200_gene_panel": overlap50,
              "overlap_count": len(overlap50),
              "top200_heg_genomewide_overlap_count": len(overlap200),
              "subsets": {}}
    for space_name, (space_truth, space_predictions) in spaces.items():
        pcc = {name: gene_pcc(space_truth, pred) for name, pred in space_predictions.items()}
        subsets = {
            "all_200_genes": np.arange(200),
            "top50_heg_in_panel_test_selected": order_test[-50:],
            "top50_heg_in_panel_train_selected": order_train[-50:],
            "bottom50_low_expression": order_test[:50],
        }
        if len(overlap_idx):
            subsets["genomewide_top50_heg_overlap"] = overlap_idx
        output["subsets"][space_name] = {
            key: {name: float(np.nanmean(pcc[name][idx])) for name in space_predictions}
            for key, idx in subsets.items()
        }
        if space_name == "raw":
            raw_pcc_ressat = pcc["ResSAT"]

    # ---- 4) 逐基因对比：原始 vs 重建空间（ResSAT）----
    recon_pcc_ressat = gene_pcc(truth_recon, recon["ResSAT"])
    expression = truth.mean(axis=0)

    (ROOT / OUT_JSON).write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))
    print("\nResSAT per-gene PCC: raw %.4f -> recon %.4f" % (
        float(np.nanmean(raw_pcc_ressat)), float(np.nanmean(recon_pcc_ressat))))

    # ---- 图 1：口径差异分解（ResSAT 一张模型预测，三个口径）----
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.6))
    fig.subplots_adjust(left=.055, right=.975, top=.88, bottom=.16, wspace=.28)

    ax = axes[0]
    threshold = np.sort(expression)[-50]
    top50 = expression >= threshold
    ax.scatter(expression[~top50], raw_pcc_ressat[~top50], s=26, color="#C9C9C9",
               label="其余 150 个基因", rasterized=True)
    ax.scatter(expression[top50], raw_pcc_ressat[top50], s=38, color=RED,
               label="前 50 个高表达基因", rasterized=True)
    ax.axvline(threshold, color=GOLD, ls="--", lw=1.4)
    ax.set_xlabel("平均表达（面板内 log1p）", fontsize=13)
    ax.set_ylabel("逐基因 PCC（直接计算）", fontsize=13)
    ax.set_title("① 旧基因集：低表达基因 PCC≈0，\n把 200 个基因平均拉到 0.087", fontsize=14, color=INK)
    ax.legend(fontsize=11, frameon=False)
    ax.tick_params(labelsize=11)

    ax = axes[1]
    ax.scatter(expression, recon_pcc_ressat, s=30, color=BLUE, rasterized=True)
    ax.set_xlabel("平均表达（面板内 log1p）", fontsize=13)
    ax.set_ylabel("逐基因 PCC（PCA 去噪后）", fontsize=13)
    ax.set_title("② 论文的计算方式：整条曲线上移，\n全基因平均 0.087 → %.3f" % np.nanmean(recon_pcc_ressat),
                 fontsize=14, color=INK)
    ax.tick_params(labelsize=11)

    ax = axes[2]
    labels = ["直接计算\n全 200 基因", "去噪后\n全 200 基因", "去噪后\n前 50 高表达基因"]
    values = [np.nanmean(raw_pcc_ressat), np.nanmean(recon_pcc_ressat),
              np.nanmean(recon_pcc_ressat[np.argsort(expression)[-50:]])]
    colors = [GRAY, BLUE, RED]
    bars = ax.bar(labels, values, color=colors, width=.6)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, value + .012, f"{value:.3f}",
                ha="center", fontsize=14, weight="bold")
    ax.set_ylim(0, max(values) * 1.22)
    ax.set_ylabel("ResSAT 平均 PCC", fontsize=13)
    ax.set_title("③ 同一批预测，仅换计算方式\n（论文：先 PCA 去噪，再看前 50 高表达基因）", fontsize=14, color=INK)
    ax.tick_params(labelsize=12)
    fig.savefig(ASSETS / "cause_decomposition.png", dpi=185, facecolor="white")
    plt.close(fig)

    # ---- 图 2：全方法 × 口径对比 ----
    names = [name for name, *_ in METHODS]
    raw_all = output["subsets"]["raw"]["all_200_genes"]
    recon_all = output["subsets"]["pca50_reconstructed"]["all_200_genes"]
    recon_top50 = output["subsets"]["pca50_reconstructed"]["top50_heg_in_panel_test_selected"]
    x = np.arange(len(names))
    width = .26
    fig, ax = plt.subplots(figsize=(12.4, 4.2))
    ax.bar(x - width, [raw_all[n] for n in names], width, color=GRAY,
           label="直接计算 · 200 个基因（旧方式）")
    ax.bar(x, [recon_all[n] for n in names], width, color=BLUE,
           label="PCA 去噪后 · 200 个基因")
    ax.bar(x + width, [recon_top50[n] for n in names], width, color=RED,
           label="PCA 去噪后 · 前 50 高表达基因（论文方式）")
    for i, name in enumerate(names):
        for offset, source in [(-width, raw_all), (0, recon_all), (width, recon_top50)]:
            value = source[name]
            ax.text(i + offset, value + .012, f"{value:.3f}", ha="center", fontsize=11.5)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=14)
    ax.set_ylabel("平均 PCC", fontsize=13)
    ax.set_ylim(0, max(recon_top50.values()) * 1.28)
    ax.legend(fontsize=13, frameon=False, ncol=3)
    ax.grid(axis="y", alpha=.25)
    ax.set_title("同一张测试切片 D1 · 同一批预测换三种计算方式重算（未重新训练）",
                 fontsize=15, color=INK)
    fig.savefig(ASSETS / "heg_benchmark_bars.png", dpi=185, facecolor="white")
    plt.close(fig)
    print("figures saved")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
