"""HEG 中间结果：用已完成的四个方法生成阶段汇报 PPT 素材（GenAR/Stem 训练中，预留）。

不训练。输出：
- results/heg_benchmark_interim.json
- figures/benchmark_ppt/cause_decomposition_v2.png   三面板原因量化
- figures/benchmark_ppt/heg_improvement_bars.png     旧面板 vs HEG 面板柱状图
- figures/benchmark_ppt/heg_topgene_heatmaps.png     最高表达基因空间热图（真值 vs 四法）

运行：py -3.12 scripts/46_heg_interim_figures.py
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
OUT_JSON = RESULTS / "heg_benchmark_interim.json"

plt.rcParams.update({"font.family": "Microsoft YaHei", "axes.unicode_minus": False})
RED, BLUE, GOLD, INK, GRAY = "#B71C1C", "#6C8EBF", "#B08D57", "#262626", "#9A9A9A"

OLD_PANEL = ROOT / "data/processed/gse240429/arrays"
HEG_PANEL = ROOT / "data/processed/gse240429_heg/arrays"

AVAILABLE = [
    ("Image Ridge", "unified_image_baselines_heg/image_ridge_C73_D1.npz", "predicted", "log"),
    ("MLP 基线", "unified_image_baselines_heg/stnet_style_mlp_C73_D1.npz", "predicted", "log"),
    ("BLEEP", "unified_image_baselines_heg/bleep_C73_D1.npz", "predicted", "log"),
    ("ResSAT", "ressat_unified_heg/unified_batch16_predictions.npz", "predicted", "log"),
]
PENDING = [("GenAR", "训练中，完成后补入"), ("Stem", "排队中，完成后补入")]


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
    u, s, vt = np.linalg.svd((train_matrix - mean) / std, full_matrices=False)
    return mean, std, vt[:50]


def reconstruct(matrix, mean, std, components):
    scores = (matrix - mean) / std @ components.T
    return mean + std * (scores @ components)


def main():
    # ---- HEG 面板：真值与四法预测 ----
    with np.load(HEG_PANEL / "C73_D1.npz") as data:
        raw_heg, genes_heg, xy = data["raw_counts"], data["genes"], data["coordinates_xy"]
    truth = panel_normalize(raw_heg)
    predictions = {}
    for name, relative, key, kind in AVAILABLE:
        with np.load(RESULTS / relative) as data:
            values = data[key].astype(np.float64)
        predictions[name] = panel_normalize(values if kind == "count" else np.expm1(values))

    train_heg = np.concatenate(
        [panel_normalize(np.load(HEG_PANEL / f"C73_{s}.npz")["raw_counts"]) for s in ("A1", "B1")],
        axis=0)
    mean, std, components = fit_pca50(train_heg)
    truth_recon = reconstruct(truth, mean, std, components)
    recon = {n: reconstruct(p, mean, std, components) for n, p in predictions.items()}
    top50 = np.argsort(train_heg.mean(axis=0))[-50:]

    raw_pcc = {n: gene_pcc(truth, p) for n, p in predictions.items()}
    recon_pcc = {n: gene_pcc(truth_recon, p) for n, p in recon.items()}

    table = {
        name: {
            "old_panel_raw_all200": None,  # 下面从旧审计补
            "heg_raw_all200": float(np.nanmean(raw_pcc[name])),
            "heg_recon_all200": float(np.nanmean(recon_pcc[name])),
            "heg_recon_top50": float(np.nanmean(recon_pcc[name][top50])),
        }
        for name, *_ in AVAILABLE
    }
    old_audit = json.loads((RESULTS / "benchmark_ppt_audit.json").read_text(encoding="utf-8"))
    for name in table:
        old_key = {"MLP 基线": "MLP 基线", "Image Ridge": "Image Ridge",
                   "BLEEP": "BLEEP", "ResSAT": "ResSAT"}[name]
        table[name]["old_panel_raw_all200"] = old_audit["stats"][old_key]["pcc"]

    output = {
        "protocol": "GSE240429 C73_D1；旧面板=200 高变基因；HEG 面板=训练集平均表达 top-200",
        "available": table,
        "pending": {n: s for n, s in PENDING},
        "note": "GenAR/Stem 正在 HEG 面板上训练，完成后补入本表与柱状图。",
    }
    (ROOT / OUT_JSON).write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))

    names = [n for n, *_ in AVAILABLE]

    # ---- 图 1：三面板原因量化 ----
    with np.load(OLD_PANEL / "C73_D1.npz") as data:
        raw_old = data["raw_counts"]
    truth_old = panel_normalize(raw_old)
    with np.load(RESULTS / "ressat_unified/unified_batch16_predictions.npz") as data:
        pred_old = panel_normalize(np.expm1(data["predicted"].astype(np.float64)))
    pcc_old_raw = gene_pcc(truth_old, pred_old)
    train_old = np.concatenate(
        [panel_normalize(np.load(OLD_PANEL / f"C73_{s}.npz")["raw_counts"]) for s in ("A1", "B1")],
        axis=0)
    mean_old, std_old, comp_old = fit_pca50(train_old)
    pcc_old_recon = gene_pcc(
        reconstruct(truth_old, mean_old, std_old, comp_old),
        reconstruct(pred_old, mean_old, std_old, comp_old))

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.5))
    fig.subplots_adjust(left=.055, right=.975, top=.86, bottom=.17, wspace=.3)
    ax = axes[0]
    expr_old = truth_old.mean(axis=0)
    thr = np.sort(expr_old)[-50]
    top = expr_old >= thr
    ax.scatter(expr_old[~top], pcc_old_raw[~top], s=22, color="#C9C9C9", rasterized=True)
    ax.scatter(expr_old[top], pcc_old_raw[top], s=34, color=RED, rasterized=True)
    ax.axvline(thr, color=GOLD, ls="--", lw=1.3)
    ax.set_xlabel("平均表达", fontsize=12)
    ax.set_ylabel("逐基因 PCC", fontsize=12)
    ax.set_title(f"① 旧基因集（按波动大选 200 个）\n低表达基因噪声大，200 个基因平均 = {np.nanmean(pcc_old_raw):.3f}",
                 fontsize=13, color=INK)
    ax.tick_params(labelsize=10)

    ax = axes[1]
    ax.scatter(truth.mean(axis=0), raw_pcc["ResSAT"], s=32, color=BLUE, rasterized=True)
    ax.set_xlabel("平均表达", fontsize=12)
    ax.set_ylabel("逐基因 PCC", fontsize=12)
    ax.set_title(f"② 高表达基因集（按表达量选 200 个）\nResSAT 200 个基因平均 = {np.nanmean(raw_pcc['ResSAT']):.3f}",
                 fontsize=13, color=INK)
    ax.tick_params(labelsize=10)

    ax = axes[2]
    labels = ["旧基因集\n直接计算", "旧基因集\nPCA 去噪后", "高表达基因集\n去噪后 · 前 50 个"]
    values = [np.nanmean(pcc_old_raw), np.nanmean(pcc_old_recon),
              table["ResSAT"]["heg_recon_top50"]]
    colors = [GRAY, BLUE, RED]
    bars = ax.bar(labels, values, color=colors, width=.58)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + .012, f"{value:.3f}",
                ha="center", fontsize=13, weight="bold")
    ax.set_ylim(0, max(values) * 1.25)
    ax.set_ylabel("ResSAT 平均 PCC", fontsize=12)
    ax.set_title("③ 同一个模型，只换基因集与计算方式", fontsize=13, color=INK)
    ax.tick_params(labelsize=10.5)
    fig.savefig(ASSETS / "cause_decomposition_v2.png", dpi=185, facecolor="white")
    plt.close(fig)

    # ---- 图 2：旧 vs HEG 柱状图 ----
    fig, ax = plt.subplots(figsize=(12.2, 3.9))
    x = np.arange(len(names))
    width = .26
    ax.bar(x - width, [table[n]["old_panel_raw_all200"] for n in names], width,
           color=GRAY, label="旧基因集 · 直接计算")
    ax.bar(x, [table[n]["heg_raw_all200"] for n in names], width,
           color=BLUE, label="高表达基因集 · 直接计算")
    ax.bar(x + width, [table[n]["heg_recon_top50"] for n in names], width,
           color=RED, label="高表达基因集 · 去噪后前 50 个")
    for i, name in enumerate(names):
        for offset, key in [(-width, "old_panel_raw_all200"), (0, "heg_raw_all200"),
                            (width, "heg_recon_top50")]:
            v = table[name][key]
            ax.text(i + offset, v + .01, f"{v:.3f}", ha="center", fontsize=10.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=13.5)
    ax.set_ylabel("平均 PCC", fontsize=12)
    ax.set_ylim(0, max(table[n]["heg_recon_top50"] for n in names) * 1.28)
    ax.legend(fontsize=12, frameon=False, ncol=3)
    ax.grid(axis="y", alpha=.25)
    ax.set_title("同一张测试切片 C73_D1：基准基因换成高表达基因后的变化",
                 fontsize=14.5, color=INK)
    fig.savefig(ASSETS / "heg_improvement_bars.png", dpi=185, facecolor="white")
    plt.close(fig)

    # ---- 图 3：最高表达基因空间热图（1×5）----
    gene_index = int(np.argmax(truth.mean(axis=0)))
    gene = str(genes_heg[gene_index])
    entries = [("实测真值", truth, None)] + [
        (name, predictions[name], raw_pcc[name][gene_index]) for name in names]
    vmax = max(float(v[:, gene_index].max()) for _, v, _ in entries)
    fig, axes = plt.subplots(1, 5, figsize=(12.6, 2.9))
    fig.subplots_adjust(left=.01, right=.97, bottom=.08, top=.88, wspace=.06)
    for ax, (label, values, pcc) in zip(axes, entries):
        dots = ax.scatter(xy[:, 0], xy[:, 1], c=values[:, gene_index], s=5, cmap="magma",
                          vmin=0, vmax=vmax, linewidths=0, rasterized=True)
        subtitle = "" if pcc is None else f"PCC = {pcc:.3f}"
        ax.set_title(f"{label}\n{subtitle}", fontsize=12.5, pad=2, color=INK)
        ax.set(xlim=(xy[:, 0].min() - 400, xy[:, 0].max() + 400),
               ylim=(xy[:, 1].max() + 400, xy[:, 1].min() - 400), aspect="equal")
        ax.axis("off")
    fig.text(.985, .52, f"统一色标\n基因 {gene}\n亮=表达高", fontsize=10.5,
             ha="center", color=INK)
    fig.savefig(ASSETS / "heg_topgene_heatmaps.png", dpi=180, facecolor="white")
    plt.close(fig)
    print("figures saved; top gene:", gene)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
