"""生成各方法在测试切片上的基因空间预测对比图 + 稀疏性统计。

对 3 个标志基因（GLUL、CYP1A2、IGKC）画 真值 vs 各方法预测 的空间热图，
并输出各方法在统一 log 空间中的零值比例误差。

输入（全部在统一 log1p(1e4) 空间或可转换）：
- results/nonimage_baselines/*_C73_D1.npz            (predicted/true/...)
- results/unified_image_baselines/*_C73_D1.npz
- results/ressat_unified/unified_batch16_predictions.npz
- results/genar_gse240429/C73_D1_predictions.npz     (predicted_counts/true_counts)
- results/stem_adapted/stem_samples_D1.npz           (mean_prediction)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
ARRAYS = ROOT / "data/processed/gse240429/arrays"
FIGURES = ROOT / "figures"

MARKER_GENES = ["GLUL", "CYP1A2", "IGKC"]

# (显示名, npz 路径, 预测字段, 真值字段, 基因字段, 是否原始计数)
METHODS = [
    ("Image Ridge", "unified_image_baselines/image_ridge_C73_D1.npz", "predicted", "true", "genes", False),
    ("ST-Net style MLP", "unified_image_baselines/stnet_style_mlp_C73_D1.npz", "predicted", "true", "genes", False),
    ("BLEEP", "unified_image_baselines/bleep_C73_D1.npz", "predicted", "true", "genes", False),
    ("ResSAT unified", "ressat_unified/unified_batch16_predictions.npz", "predicted", "true", "genes", False),
    ("GenAR", "genar_gse240429/C73_D1_predictions.npz", "predicted_counts", "true_counts", "gene_names", True),
    ("Stem", "stem_adapted/stem_samples_D1.npz", "mean_prediction", "true", "genes", False),
]


def load_method(spec: tuple) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray] | None:
    name, relative, pred_key, true_key, gene_key, is_count = spec
    path = RESULTS / relative
    if not path.exists():
        print(f"跳过 {name}（缺少 {path}）")
        return None
    data = np.load(path, allow_pickle=True)
    predicted = data[pred_key].astype(np.float64)
    true = data[true_key].astype(np.float64)
    genes = [str(gene) for gene in data[gene_key]]
    if "coordinates_xy" in data.files:
        coordinates = data["coordinates_xy"].astype(np.float32)
    else:
        # GenAR 的 npz 不含坐标；所有方法的行序都与 arrays 一致，取统一坐标。
        with np.load(ARRAYS / "C73_D1.npz") as source:
            coordinates = source["coordinates_xy"].astype(np.float32)
        if len(coordinates) != len(predicted):
            raise ValueError(f"{name} 行数与统一坐标不一致")
    if is_count:
        def log_normalize(counts: np.ndarray) -> np.ndarray:
            library = counts.sum(axis=1, keepdims=True)
            library[library <= 0] = 1.0
            return np.log1p(counts / library * 1e4)

        predicted = log_normalize(predicted)
        true = log_normalize(true)
    return predicted, true, genes, coordinates


def zero_fraction_error(true: np.ndarray, predicted: np.ndarray) -> dict:
    true_zero = float(np.mean(true == 0))
    predicted_zero = float(np.mean(predicted == 0))
    return {
        "true_zero_fraction": true_zero,
        "predicted_zero_fraction": predicted_zero,
        "absolute_zero_fraction_error": abs(predicted_zero - true_zero),
    }


def main() -> None:
    sparsity: dict[str, dict] = {}
    rows = []
    for spec in METHODS:
        loaded = load_method(spec)
        if loaded is None:
            continue
        predicted, true, genes, coordinates = loaded
        rows.append((spec[0], predicted, true, genes, coordinates))
        sparsity[spec[0]] = zero_fraction_error(true, predicted)
        print(f"{spec[0]}: zero err {sparsity[spec[0]]['absolute_zero_fraction_error']:.4f}")

    if not rows:
        raise RuntimeError("没有可用预测，请先完成实验")

    # 第一行是真值；后面每方法一行。
    first_genes = rows[0][3]
    figure, axes = plt.subplots(
        len(rows) + 1,
        len(MARKER_GENES),
        figsize=(4.2 * len(MARKER_GENES), 3.1 * (len(rows) + 1)),
        constrained_layout=True,
    )
    for column, gene in enumerate(MARKER_GENES):
        for gene_index in range(len(first_genes)):
            if first_genes[gene_index] == gene:
                break
        else:
            raise ValueError(f"{gene} 不在基因面板中")

        # 真值行
        axis = axes[0, column]
        values = rows[0][2][:, gene_index]
        axis.scatter(
            rows[0][4][:, 0], rows[0][4][:, 1], c=values, s=9,
            cmap="magma", linewidths=0,
            vmin=0, vmax=float(np.percentile(values, 98)),
        )
        axis.set_title(f"TRUE  {gene}" if column == 0 else f"TRUE\n{gene}")
        axis.invert_yaxis()
        axis.set_aspect("equal")
        axis.set_xticks([])
        axis.set_yticks([])

        for row_index, (name, predicted, _, genes_row, coordinates_row) in enumerate(rows, start=1):
            axis = axes[row_index, column]
            values = predicted[:, gene_index]
            pcc = float(np.corrcoef(rows[0][2][:, gene_index], values)[0, 1])
            axis.scatter(
                coordinates_row[:, 0], coordinates_row[:, 1], c=values, s=9,
                cmap="magma", linewidths=0,
                vmin=0, vmax=float(np.percentile(values, 98)),
            )
            title = f"{name}\nPCC={pcc:.3f}" if column == 0 else f"PCC={pcc:.3f}"
            axis.set_title(title, fontsize=9)
            axis.invert_yaxis()
            axis.set_aspect("equal")
            axis.set_xticks([])
            axis.set_yticks([])

    figure.suptitle(
        "Predicted vs measured spatial expression on held-out slide C73_D1 "
        "(unified log1p(1e4) scale)",
        fontsize=13,
    )
    figure.savefig(FIGURES / "05_gene_spatial_predictions.png", dpi=150)
    plt.close(figure)

    (RESULTS / "sparsity_summary.json").write_text(
        json.dumps(sparsity, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"完成：{FIGURES / '05_gene_spatial_predictions.png'}")
    print(f"完成：{RESULTS / 'sparsity_summary.json'}")


if __name__ == "__main__":
    main()
