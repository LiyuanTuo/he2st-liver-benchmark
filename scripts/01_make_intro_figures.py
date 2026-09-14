"""用真实空间转录组标签画出“模型要预测什么”。

这个脚本暂时不需要 H&E 大图，因此可以在 GEO 原始包下载期间运行。
它读取 BLEEP 官方仓库随附的 GSE240429 第 1 张切片数据，完成三项检查：
1. 表达矩阵的列必须与 barcode 文件一一对应；
2. 每个 barcode 必须能在空间坐标表中找到；
3. 基因名称必须与 HVG 布尔掩码保持相同顺序。

输出：
  figures/01_spatial_transcriptomics_examples.png
  figures/02_expression_matrix_example.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# 无论从哪个工作目录启动，都从脚本位置定位项目根目录。
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "third_party/BLEEP/GSE240429_data/data"
FIGURES = ROOT / "figures"


def load_slide_one() -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """返回表达矩阵 [spot, gene]、基因名和按表达顺序排列的坐标。"""

    expression_root = DATA / "filtered_expression_matrices"

    # hvg_union 与原始 36,601 个 feature 同序；True 表示该基因被保留。
    hvg_mask = np.load(expression_root / "hvg_union.npy")
    features = pd.read_csv(
        expression_root / "1/features.tsv",
        sep="\t",
        header=None,
        names=["gene_id", "gene_name", "feature_type"],
    )
    if len(features) != len(hvg_mask):
        raise ValueError("features.tsv 与 hvg_union.npy 长度不一致")
    genes = features.loc[hvg_mask, "gene_name"].tolist()

    # 作者保存的是 [gene, spot]，这里转成更直观的 [spot, gene]。
    expression = np.load(expression_root / "1/hvg_matrix.npy").T
    barcodes = pd.read_csv(
        expression_root / "1/barcodes.tsv",
        sep="\t",
        header=None,
        names=["barcode"],
    )["barcode"].tolist()
    if expression.shape != (len(barcodes), len(genes)):
        raise ValueError(
            "表达矩阵形状不正确："
            f"matrix={expression.shape}, barcodes={len(barcodes)}, genes={len(genes)}"
        )

    # 10x 坐标列依次为 barcode、是否在组织内、阵列行列、图像像素行列。
    positions = pd.read_csv(
        DATA / "tissue_pos_matrices/tissue_positions_list_1.csv",
        header=None,
        names=["barcode", "in_tissue", "array_row", "array_col", "pixel_y", "pixel_x"],
    ).set_index("barcode")
    missing = sorted(set(barcodes) - set(positions.index))
    if missing:
        raise ValueError(f"{len(missing)} 个表达 barcode 没有空间坐标")

    # 关键：用表达矩阵的 barcode 顺序重排坐标，禁止按文件当前行号假定对齐。
    aligned_positions = positions.loc[barcodes].reset_index()
    if not np.all(aligned_positions["in_tissue"].to_numpy() == 1):
        raise ValueError("表达矩阵中出现了组织外 spot")
    return expression, genes, aligned_positions


def draw_spatial_examples(
    expression: np.ndarray,
    genes: list[str],
    positions: pd.DataFrame,
) -> None:
    """画出 spot 网格和三个具有不同空间模式的真实基因。"""

    selected_genes = ["GLUL", "CYP1A2", "IGKC"]
    missing = [gene for gene in selected_genes if gene not in genes]
    if missing:
        raise ValueError(f"示例基因不在当前面板：{missing}")

    figure, axes = plt.subplots(1, 4, figsize=(16, 4.3), constrained_layout=True)
    x = positions["pixel_x"].to_numpy()
    y = positions["pixel_y"].to_numpy()

    axes[0].scatter(x, y, s=8, color="#53657D", linewidths=0)
    axes[0].set_title(f"Visium spot locations\n{len(x):,} spots")
    axes[0].set_facecolor("#F5F2EC")

    for axis, gene in zip(axes[1:], selected_genes):
        values = expression[:, genes.index(gene)]
        points = axis.scatter(x, y, c=values, s=10, cmap="magma", linewidths=0)
        axis.set_title(f"Measured expression: {gene}")
        figure.colorbar(points, ax=axis, shrink=0.76, label="log-normalized expression")

    for axis in axes:
        # 图像坐标的 y 轴向下增长，所以需要翻转才能与原始 H&E 方向一致。
        axis.invert_yaxis()
        axis.set_aspect("equal")
        axis.set_xticks([])
        axis.set_yticks([])

    figure.suptitle(
        "One ST slide contains many gene maps — the model predicts all columns of the matrix",
        fontsize=14,
    )
    figure.savefig(FIGURES / "01_spatial_transcriptomics_examples.png", dpi=220)
    plt.close(figure)


def draw_matrix_example(expression: np.ndarray, genes: list[str]) -> None:
    """画出 spot×gene 矩阵，强调最终输出不是单张热图。"""

    # 只为显示选取方差最大的 40 个基因；这不是训练或测试的基因选择。
    top_gene_indices = np.argsort(expression.var(axis=0))[-40:]
    shown = expression[:180, top_gene_indices]
    shown_genes = [genes[index] for index in top_gene_indices]

    figure, axis = plt.subplots(figsize=(13, 5), constrained_layout=True)
    image = axis.imshow(shown.T, aspect="auto", cmap="viridis", interpolation="nearest")
    axis.set_title("A small view of the measured spot × gene matrix")
    axis.set_xlabel("180 example spots (matrix rows)")
    axis.set_ylabel("40 high-variance genes (matrix columns)")
    axis.set_yticks(range(0, len(shown_genes), 4))
    axis.set_yticklabels(shown_genes[::4], fontsize=8)
    figure.colorbar(image, ax=axis, label="log-normalized expression")
    figure.savefig(FIGURES / "02_expression_matrix_example.png", dpi=220)
    plt.close(figure)


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    expression, genes, positions = load_slide_one()
    draw_spatial_examples(expression, genes, positions)
    draw_matrix_example(expression, genes)
    print(f"Validated matrix: {expression.shape[0]:,} spots × {expression.shape[1]:,} genes")
    print(f"Saved figures to: {FIGURES}")


if __name__ == "__main__":
    main()
