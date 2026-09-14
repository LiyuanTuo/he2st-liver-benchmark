"""把 BLEEP/GSE240429 原始表达整理成统一、无测试泄漏的数据。

固定主划分：
  train = C73_A1, C73_B1
  validation = C73_C1
  test = C73_D1

基因选择、表达方差排序和 GenAR 基因层次排序只使用两张训练切片。
输出同时服务于本项目的回归模型和官方 GenAR 模型主体。
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "third_party/BLEEP/GSE240429_data/data"
OUTPUT = ROOT / "data/processed/gse240429"

SLIDES = {
    "C73_A1": "1",
    "C73_B1": "2",
    "C73_C1": "3",
    "C73_D1": "4",
}
SPLIT = {
    "train": ["C73_A1", "C73_B1"],
    "validation": ["C73_C1"],
    "test": ["C73_D1"],
}
GENE_COUNT = 200
LIBRARY_SCALE = 10_000.0


def sha256_text(lines: list[str]) -> str:
    """对有顺序的字符串列表计算稳定哈希。"""

    return hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()


def load_feature_table() -> pd.DataFrame:
    """读取 10x feature 表，并确认四张切片使用同一基因顺序。"""

    reference: pd.DataFrame | None = None
    for number in SLIDES.values():
        current = pd.read_csv(
            SOURCE / f"filtered_expression_matrices/{number}/features.tsv",
            sep="\t",
            header=None,
            names=["gene_id", "gene_name", "feature_type"],
        )
        if reference is None:
            reference = current
        elif not current.equals(reference):
            raise ValueError(f"切片 {number} 的 features.tsv 顺序不同")
    assert reference is not None
    return reference


def load_slide(number: str) -> tuple[sparse.csr_matrix, list[str], np.ndarray]:
    """读取一张切片，按表达 barcode 顺序返回计数、barcode 和 xy 坐标。"""

    expression_dir = SOURCE / f"filtered_expression_matrices/{number}"
    counts = mmread(expression_dir / "matrix.mtx").tocsr().T
    if counts.data.size and not np.allclose(counts.data, np.rint(counts.data)):
        raise ValueError(f"切片 {number} 不是原始整数计数")
    counts = counts.astype(np.int32)

    barcodes = pd.read_csv(
        expression_dir / "barcodes.tsv", sep="\t", header=None
    )[0].astype(str).tolist()
    if counts.shape[0] != len(barcodes):
        raise ValueError(f"切片 {number} 的表达行与 barcode 数不一致")

    positions = pd.read_csv(
        SOURCE / f"tissue_pos_matrices/tissue_positions_list_{number}.csv",
        header=None,
        names=["barcode", "in_tissue", "array_row", "array_col", "pixel_y", "pixel_x"],
    ).set_index("barcode")
    missing = set(barcodes) - set(positions.index)
    if missing:
        raise ValueError(f"切片 {number} 有 {len(missing)} 个 barcode 缺少坐标")
    positions = positions.loc[barcodes]
    if not np.all(positions["in_tissue"].to_numpy() == 1):
        raise ValueError(f"切片 {number} 的表达矩阵包含组织外 spot")

    # AnnData 和 GenAR 约定坐标顺序为 (x, y)，而 10x CSV 是先 y 后 x。
    coordinates_xy = positions[["pixel_x", "pixel_y"]].to_numpy(np.float32)
    return counts, barcodes, coordinates_xy


def log_normalize(counts: sparse.csr_matrix) -> sparse.csr_matrix:
    """按 spot 总计数缩放到 1e4 后做 log1p，不修改原始计数。"""

    library_size = np.asarray(counts.sum(axis=1)).ravel().astype(np.float64)
    if np.any(library_size <= 0):
        raise ValueError("发现总计数为 0 的 spot")
    normalized = counts.astype(np.float64).multiply(
        (LIBRARY_SCALE / library_size)[:, None]
    ).tocsr()
    normalized.data = np.log1p(normalized.data)
    return normalized.astype(np.float32)


def select_training_genes(
    training_counts: sparse.csr_matrix,
    features: pd.DataFrame,
) -> tuple[np.ndarray, list[str], dict[str, float]]:
    """只用训练切片选 200 个高变基因，并返回原 feature 列索引。"""

    normalized = log_normalize(training_counts)
    means = np.asarray(normalized.mean(axis=0)).ravel()
    second_moment = np.asarray(normalized.power(2).mean(axis=0)).ravel()
    variances = np.maximum(second_moment - means**2, 0)
    detected_fraction = np.asarray((training_counts > 0).mean(axis=0)).ravel()

    names = features["gene_name"].fillna("").astype(str)
    unique_name = ~names.duplicated(keep="first")
    technical = names.str.match(r"^(MT-|RPS\d|RPL\d)", case=False)
    eligible = (
        unique_name.to_numpy()
        & ~technical.to_numpy()
        & (names.str.len().to_numpy() > 0)
        & (detected_fraction >= 0.05)
    )
    eligible_indices = np.flatnonzero(eligible)
    if len(eligible_indices) < GENE_COUNT:
        raise ValueError(f"只有 {len(eligible_indices)} 个基因通过训练集过滤")

    selected = eligible_indices[
        np.argsort(variances[eligible_indices])[-GENE_COUNT:][::-1]
    ]
    selected_names = names.iloc[selected].tolist()
    if len(set(selected_names)) != GENE_COUNT:
        raise ValueError("选择后的基因名不唯一")

    diagnostics = {
        "eligible_gene_count": int(len(eligible_indices)),
        "minimum_detection_fraction": float(detected_fraction[selected].min()),
        "minimum_selected_variance": float(variances[selected].min()),
        "maximum_selected_variance": float(variances[selected].max()),
    }
    return selected, selected_names, diagnostics


def apply_genar_hierarchy(
    training_counts: sparse.csr_matrix,
    selected_indices: np.ndarray,
    selected_names: list[str],
) -> tuple[np.ndarray, list[str]]:
    """调用官方 GenAR 的两阶段 KMeans，实现相同的粗到细基因排序。"""

    genar_src = ROOT / "third_party/GenAR/src"
    sys.path.insert(0, str(genar_src))
    from preprocess.gene_clustering import GeneClusteringProcessor

    dense_selected_counts = training_counts[:, selected_indices].toarray()
    processor = GeneClusteringProcessor(random_state=42)
    permutation = processor._perform_clustering(dense_selected_counts)
    ordered_indices = selected_indices[permutation]
    ordered_names = [selected_names[index] for index in permutation]
    return ordered_indices, ordered_names


def save_slide(
    slide_id: str,
    counts: sparse.csr_matrix,
    barcodes: list[str],
    coordinates_xy: np.ndarray,
    gene_indices: np.ndarray,
    genes: list[str],
    features: pd.DataFrame,
) -> dict[str, object]:
    """保存原始计数 h5ad，以及模型/绘图都能直接读取的压缩数组。"""

    selected_counts = counts[:, gene_indices].tocsr()
    # 标准化分母必须是该 spot 的全转录组 library size，而不是仅 200 个目标
    # 基因之和；后者会让某些 spot 分母为 0，也会扭曲不同切片间的尺度。
    normalized = log_normalize(counts)[:, gene_indices].toarray()
    total_counts_all_genes = np.asarray(counts.sum(axis=1)).ravel().astype(np.int64)

    adata = ad.AnnData(
        X=selected_counts,
        obs=pd.DataFrame(
            {"total_counts_all_genes": total_counts_all_genes},
            index=pd.Index(barcodes, name="barcode"),
        ),
        var=pd.DataFrame(
            {
                "gene_id": features.iloc[gene_indices]["gene_id"].to_numpy(),
                "gene_name": genes,
            },
            index=pd.Index(genes, name="gene_name"),
        ),
    )
    adata.obsm["spatial"] = coordinates_xy
    adata.uns["source"] = "NCBI GEO GSE240429; expression/coordinates from BLEEP repo"
    adata.uns["expression_scale"] = "raw non-negative integer counts"
    adata.write_h5ad(OUTPUT / "st" / f"{slide_id}.h5ad", compression="gzip")

    np.savez_compressed(
        OUTPUT / "arrays" / f"{slide_id}.npz",
        raw_counts=selected_counts.toarray().astype(np.int32),
        log_normalized=normalized.astype(np.float32),
        coordinates_xy=coordinates_xy,
        barcodes=np.asarray(barcodes),
        genes=np.asarray(genes),
    )
    return {
        "spots": int(selected_counts.shape[0]),
        "genes": int(selected_counts.shape[1]),
        "total_counts": int(selected_counts.sum()),
        "zero_fraction": float(1.0 - selected_counts.nnz / np.prod(selected_counts.shape)),
    }


def main() -> None:
    (OUTPUT / "st").mkdir(parents=True, exist_ok=True)
    (OUTPUT / "arrays").mkdir(parents=True, exist_ok=True)
    (OUTPUT / "processed_data").mkdir(parents=True, exist_ok=True)

    features = load_feature_table()
    loaded = {slide: load_slide(number) for slide, number in SLIDES.items()}
    if any(item[0].shape[1] != len(features) for item in loaded.values()):
        raise ValueError("至少一张切片的表达基因数与 feature 表不一致")

    training_counts = sparse.vstack(
        [loaded[slide][0] for slide in SPLIT["train"]], format="csr"
    )
    selected_indices, selected_names, diagnostics = select_training_genes(
        training_counts, features
    )
    gene_indices, genes = apply_genar_hierarchy(
        training_counts, selected_indices, selected_names
    )

    slide_summary = {}
    for slide_id, (counts, barcodes, coordinates) in loaded.items():
        slide_summary[slide_id] = save_slide(
            slide_id,
            counts,
            barcodes,
            coordinates,
            gene_indices,
            genes,
            features,
        )

    processed_data = OUTPUT / "processed_data"
    (processed_data / "all_slide_lst.txt").write_text(
        "\n".join(SLIDES) + "\n", encoding="utf-8"
    )
    (processed_data / "selected_gene_list.txt").write_text(
        "\n".join(genes) + "\n", encoding="utf-8"
    )
    (processed_data / "unclustered_selected_gene_list.txt").write_text(
        "\n".join(selected_names) + "\n", encoding="utf-8"
    )

    manifest = {
        "schema_version": 1,
        "source": "GSE240429",
        "slides": slide_summary,
        "split": SPLIT,
        "gene_selection": {
            "fit_on_slides": SPLIT["train"],
            "method": "top variance after per-spot 1e4 log1p; detected in >=5% train spots",
            "excluded": "duplicated symbols, mitochondrial, RPS/RPL ribosomal genes",
            "gene_count": GENE_COUNT,
            "ordered_gene_sha256": sha256_text(genes),
            "genar_hierarchy": "official GenAR two-stage KMeans, random_state=42",
            **diagnostics,
        },
    }
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (OUTPUT / "split.json").write_text(
        json.dumps(SPLIT, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"Prepared {len(SLIDES)} slides in {OUTPUT}")
    print(f"Split: {SPLIT}")
    print(f"Ordered genes SHA256: {sha256_text(genes)}")


if __name__ == "__main__":
    main()
