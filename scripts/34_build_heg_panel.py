"""构建 HEG 面板：按训练切片平均表达选 top-200 高表达基因，重建统一数据。

与原 200 基因面板（scripts/02，按方差选）的唯一差别是基因选择标准：
- 训练切片（A1+B1）全转录组 log1p(1e4 归一化) 表达的平均值降序取前 200；
- 不再使用方差排序、不再剔除线粒体/RPS/RPL（高表达基因按其定义保留）；
- 仍排除重复符号与空名；仍套用官方 GenAR 两阶段 KMeans 层次排序；
- 划分、barcode 行序、坐标、图像缓存与旧面板完全一致（逐 barcode 断言）。

输出 data/processed/gse240429_heg/，旧目录不改动。
运行：wsl python3 scripts/34_build_heg_panel.py
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
OUTPUT = ROOT / "data/processed/gse240429_heg"
OLD_ARRAYS = ROOT / "data/processed/gse240429/arrays"
OLD_PROCESSED = ROOT / "data/processed/gse240429/processed_data"

SLIDES = {"C73_A1": "1", "C73_B1": "2", "C73_C1": "3", "C73_D1": "4"}
SPLIT = {"train": ["C73_A1", "C73_B1"], "validation": ["C73_C1"], "test": ["C73_D1"]}
GENE_COUNT = 200
LIBRARY_SCALE = 10_000.0


def sha256_text(lines):
    return hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()


def load_feature_table():
    reference = None
    for number in SLIDES.values():
        current = pd.read_csv(
            SOURCE / f"filtered_expression_matrices/{number}/features.tsv",
            sep="\t", header=None,
            names=["gene_id", "gene_name", "feature_type"],
        )
        if reference is None:
            reference = current
        elif not current.equals(reference):
            raise ValueError(f"切片 {number} 的 features.tsv 顺序不同")
    return reference


def load_slide(number):
    expression_dir = SOURCE / f"filtered_expression_matrices/{number}"
    counts = mmread(expression_dir / "matrix.mtx").tocsr().T.astype(np.int32)
    barcodes = pd.read_csv(
        expression_dir / "barcodes.tsv", sep="\t", header=None
    )[0].astype(str).tolist()
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
    coordinates_xy = positions[["pixel_x", "pixel_y"]].to_numpy(np.float32)
    return counts, barcodes, coordinates_xy


def log_normalize(counts):
    library_size = np.asarray(counts.sum(axis=1)).ravel().astype(np.float64)
    if np.any(library_size <= 0):
        raise ValueError("发现总计数为 0 的 spot")
    normalized = counts.astype(np.float64).multiply(
        (LIBRARY_SCALE / library_size)[:, None]
    ).tocsr()
    normalized.data = np.log1p(normalized.data)
    return normalized.astype(np.float32)


def select_heg(training_counts, features):
    """训练切片平均 log1p(1e4 归一化) 表达最高的 200 个基因。"""
    normalized = log_normalize(training_counts)
    means = np.asarray(normalized.mean(axis=0)).ravel()
    names = features["gene_name"].fillna("").astype(str)
    unique_name = ~names.duplicated(keep="first")
    eligible = unique_name.to_numpy() & (names.str.len().to_numpy() > 0)
    eligible_indices = np.flatnonzero(eligible)
    if len(eligible_indices) < GENE_COUNT:
        raise ValueError("合格基因不足 200")
    selected = eligible_indices[np.argsort(means[eligible_indices])[-GENE_COUNT:][::-1]]
    selected_names = names.iloc[selected].tolist()
    if len(set(selected_names)) != GENE_COUNT:
        raise ValueError("选择后的基因名不唯一")
    diagnostics = {
        "eligible_gene_count": int(len(eligible_indices)),
        "minimum_selected_mean_expression": float(means[selected].min()),
        "maximum_selected_mean_expression": float(means[selected].max()),
        "mean_expression_of_selected": float(means[selected].mean()),
        "mitochondrial_gene_count": int(names.iloc[selected].str.startswith("MT-").sum()),
        "ribosomal_gene_count": int(names.iloc[selected].str.match(r"^RP[SL]").sum()),
    }
    return selected, selected_names, diagnostics


def apply_genar_hierarchy(training_counts, selected_indices, selected_names):
    genar_src = ROOT / "third_party/GenAR/src"
    sys.path.insert(0, str(genar_src))
    from preprocess.gene_clustering import GeneClusteringProcessor

    dense_selected_counts = training_counts[:, selected_indices].toarray()
    processor = GeneClusteringProcessor(random_state=42)
    permutation = processor._perform_clustering(dense_selected_counts)
    ordered_indices = selected_indices[permutation]
    ordered_names = [selected_names[index] for index in permutation]
    return ordered_indices, ordered_names


def save_slide(slide_id, counts, barcodes, coordinates_xy, gene_indices, genes, features):
    selected_counts = counts[:, gene_indices].tocsr()
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
    adata.uns["source"] = "NCBI GEO GSE240429; HEG panel (top-200 by train mean expression)"
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


def main():
    for sub in ("st", "arrays", "processed_data", "processed_data/spot_features_resnet18"):
        (OUTPUT / sub).mkdir(parents=True, exist_ok=True)

    features = load_feature_table()
    loaded = {slide: load_slide(number) for slide, number in SLIDES.items()}
    if any(item[0].shape[1] != len(features) for item in loaded.values()):
        raise ValueError("表达基因数与 feature 表不一致")

    training_counts = sparse.vstack(
        [loaded[slide][0] for slide in SPLIT["train"]], format="csr"
    )
    selected_indices, selected_names, diagnostics = select_heg(training_counts, features)
    gene_indices, genes = apply_genar_hierarchy(
        training_counts, selected_indices, selected_names
    )

    slide_summary = {}
    for slide_id, (counts, barcodes, coordinates) in loaded.items():
        slide_summary[slide_id] = save_slide(
            slide_id, counts, barcodes, coordinates, gene_indices, genes, features
        )
        # 断言与旧面板 barcode 行序一致（图像块/特征缓存可复用）。
        with np.load(OLD_ARRAYS / f"{slide_id}.npz") as old:
            assert np.array_equal(old["barcodes"], np.asarray(barcodes)), slide_id
        # 复制 GenAR 需要的 spot 特征（图像级、与基因无关）。
        feature_file = OLD_PROCESSED / "spot_features_resnet18" / f"{slide_id}_resnet18.pt"
        import shutil
        shutil.copy2(feature_file, OUTPUT / "processed_data/spot_features_resnet18")

    processed_data = OUTPUT / "processed_data"
    (processed_data / "all_slide_lst.txt").write_text("\n".join(SLIDES) + "\n", encoding="utf-8")
    (processed_data / "selected_gene_list.txt").write_text("\n".join(genes) + "\n", encoding="utf-8")
    (processed_data / "unclustered_selected_gene_list.txt").write_text(
        "\n".join(selected_names) + "\n", encoding="utf-8"
    )

    manifest = {
        "schema_version": 1,
        "source": "GSE240429 (HEG panel rebuild)",
        "slides": slide_summary,
        "split": SPLIT,
        "gene_selection": {
            "fit_on_slides": SPLIT["train"],
            "method": "top-200 by mean log1p(1e4-normalized) expression on train spots; "
                      "detected genes only; duplicates removed; no MT/RPL/RPS exclusion",
            "gene_count": GENE_COUNT,
            "ordered_gene_sha256": sha256_text(genes),
            "unclustered_gene_sha256": sha256_text(selected_names),
            **diagnostics,
        },
        "notes": [
            "与 data/processed/gse240429 的区别仅在基因选择标准（高表达代替高方差）。",
            "barcode 行序与旧面板逐切片断言一致，图像块与 ResNet18 特征缓存直接复用。",
        ],
    }
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"slide_summary": slide_summary, "diagnostics": diagnostics,
                      "top10_genes": genes[:10]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
