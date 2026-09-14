"""从 10x 原文件重建 ResSAT 的 SA/SP 处理后数据。

流程严格依据 ResSAT 正式版 Methods：spot QC -> 1e4 总量归一化 -> log1p ->
按 section 选择 2,000 HVG -> scale(max=10) -> PCA-50 -> Harmony。图像按
Space Ranger 报告的 spot 直径裁成 88x88 patch。官方没有公开这段预处理代码，
所以本脚本是可审计重建，而不是作者脚本的逐字复刻。
"""

from __future__ import annotations

import csv
import hashlib
import json
import pickle
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import tifffile
import harmonypy


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/ressat_mouse_brain"
OUT = ROOT / "data/processed/ressat_original_rebuilt"
DATASETS = {"SA": ("SA1", "SA2"), "SP": ("SP1", "SP2")}


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_and_qc(sample: str) -> tuple[ad.AnnData, dict]:
    """读取一个 10x 切片，执行论文阈值，并把像素坐标写入 obs。"""
    folder = RAW / sample
    matrix_path = next(folder.glob("*_filtered_feature_bc_matrix.h5"))
    image_path = next(folder.glob("*_image.tif"))
    spatial = folder / "spatial"
    counts = sc.read_10x_h5(matrix_path)
    counts.var_names_make_unique()

    positions = pd.read_csv(
        spatial / "tissue_positions_list.csv",
        header=None,
        names=["barcode", "in_tissue", "array_row", "array_col", "pixel_y", "pixel_x"],
        index_col="barcode",
    )
    positions = positions.loc[counts.obs_names]
    total = np.asarray(counts.X.sum(axis=1)).ravel()
    detected = np.asarray((counts.X > 0).sum(axis=1)).ravel()
    mt_mask = counts.var_names.str.upper().str.startswith("MT-")
    mt = np.asarray(counts[:, mt_mask].X.sum(axis=1)).ravel() / np.maximum(total, 1)
    with (spatial / "scalefactors_json.json").open(encoding="utf-8") as source:
        scale = json.load(source)
    radius = int(float(scale["spot_diameter_fullres"]) / 2)
    with tifffile.TiffFile(image_path) as tif:
        height, width = tif.pages[0].shape[:2]
    inside = (
        (positions["pixel_x"].to_numpy() - radius >= 0)
        & (positions["pixel_x"].to_numpy() + radius <= width)
        & (positions["pixel_y"].to_numpy() - radius >= 0)
        & (positions["pixel_y"].to_numpy() + radius <= height)
    )
    keep = (total >= 1000) & (total <= 70000) & (detected >= 3) & (mt <= 0.20) & inside
    counts = counts[keep].copy()
    kept_positions = positions.iloc[np.flatnonzero(keep)]
    counts.obs["barcode_original"] = counts.obs_names.astype(str)
    counts.obs["pixel_x"] = kept_positions["pixel_x"].to_numpy(dtype=np.int32)
    counts.obs["pixel_y"] = kept_positions["pixel_y"].to_numpy(dtype=np.int32)
    counts.obs["array_row"] = kept_positions["array_row"].to_numpy(dtype=np.int32)
    counts.obs["array_col"] = kept_positions["array_col"].to_numpy(dtype=np.int32)
    counts.uns["image_path"] = str(image_path)
    counts.uns["patch_radius"] = radius
    audit = {
        "raw_spots": int(len(keep)),
        "kept_spots": int(keep.sum()),
        "raw_genes": int(counts.n_vars),
        "removed_low_umi": int((total < 1000).sum()),
        "removed_high_umi": int((total > 70000).sum()),
        "removed_mt_over_20pct": int((mt > 0.20).sum()),
        "removed_incomplete_patch": int((~inside).sum()),
        "spot_diameter_fullres": float(scale["spot_diameter_fullres"]),
        "patch_shape": [radius * 2, radius * 2, 3],
        "matrix_sha256": file_hash(matrix_path),
        "image_sha256": file_hash(image_path),
    }
    return counts, audit


def normalized_coordinates(frame: pd.DataFrame) -> np.ndarray:
    xy = frame[["pixel_x", "pixel_y"]].to_numpy(dtype=np.float64)
    low, high = xy.min(axis=0), xy.max(axis=0)
    return ((xy - low) / np.maximum(high - low, 1)).astype(np.float32)


def build_patches(sample: str, frame: pd.DataFrame, radius: int) -> list[np.ndarray]:
    """利用可内存映射 TIFF 裁图；uint8 可避免 torchvision 浮点回绕。"""
    image_path = next((RAW / sample).glob("*_image.tif"))
    image = tifffile.memmap(image_path)
    patches = []
    for x, y in frame[["pixel_x", "pixel_y"]].to_numpy(dtype=np.int32):
        patch = np.ascontiguousarray(image[y - radius : y + radius, x - radius : x + radius, :3])
        if patch.shape != (radius * 2, radius * 2, 3):
            raise ValueError(f"{sample} patch 形状异常：{patch.shape}")
        patches.append(patch.astype(np.uint8, copy=False))
    return patches


def process_dataset(name: str, samples: tuple[str, str]) -> None:
    target = OUT / name
    target.mkdir(parents=True, exist_ok=True)
    sections, audit = [], {}
    for sample in samples:
        section, audit[sample] = load_and_qc(sample)
        sections.append(section)
        print(f"{name}/{sample}: {audit[sample]['raw_spots']} -> {audit[sample]['kept_spots']} spots")

    # 两切片只保留共有基因；HVG 的 batch_key 防止单切片主导特征选择。
    merged = ad.concat(sections, join="inner", label="section", keys=list(samples), index_unique="-")
    sc.pp.normalize_total(merged, target_sum=10000)
    sc.pp.log1p(merged)
    sc.pp.highly_variable_genes(merged, n_top_genes=2000, batch_key="section")
    merged = merged[:, merged.var["highly_variable"]].copy()
    genes = merged.var_names.astype(str).tolist()

    # 保存 PCA 之前的完整 log-normalized 真值，供严格口径评测。
    log_expression = merged.X.toarray().astype(np.float32) if hasattr(merged.X, "toarray") else np.asarray(merged.X, dtype=np.float32)
    gene_mean = log_expression.mean(axis=0, dtype=np.float64)
    gene_std = log_expression.std(axis=0, ddof=1, dtype=np.float64)
    gene_std[gene_std < 1e-8] = 1.0
    sc.pp.scale(merged, max_value=10)
    sc.tl.pca(merged, n_comps=50, svd_solver="arpack", random_state=42)
    # harmonypy 2.0 已把 Z_corr 改成 cell×PC，而当前 scanpy wrapper 仍无条件转置，
    # 会产生 50×N 的错误形状；这里直接调用并兼容新旧两个返回方向。
    harmony_result = harmonypy.run_harmony(merged.obsm["X_pca"], merged.obs, "section")
    harmony = np.asarray(harmony_result.Z_corr, dtype=np.float32)
    if harmony.shape == (50, merged.n_obs):
        harmony = harmony.T
    if harmony.shape != (merged.n_obs, 50):
        raise ValueError(f"Harmony 返回形状异常：{harmony.shape}")
    components = np.asarray(merged.varm["PCs"].T, dtype=np.float32)

    with (target / "gene_list.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["gene"])
        writer.writerows([[gene] for gene in genes])
    with (target / "pca_info.pkl").open("wb") as handle:
        pickle.dump({"mean": gene_mean, "std": gene_std, "components": components, "gene_names": genes}, handle, protocol=4)

    for number, sample in enumerate(samples, start=1):
        mask = merged.obs["section"].astype(str).to_numpy() == sample
        frame = merged.obs.loc[mask]
        indices = np.flatnonzero(mask)
        radius = int(sections[number - 1].uns["patch_radius"])
        patches = build_patches(sample, frame, radius)
        records = [(patch, harmony[index]) for patch, index in zip(patches, indices)]
        coordinates = normalized_coordinates(frame)
        section_dir = target / f"Section_{number}"
        section_dir.mkdir(exist_ok=True)
        with (section_dir / "dataset.pkl").open("wb") as handle:
            pickle.dump(records, handle, protocol=4)
        with (section_dir / "locations.pkl").open("wb") as handle:
            pickle.dump([tuple(map(float, row)) for row in coordinates], handle, protocol=4)
        np.savez_compressed(
            section_dir / "strict_truth.npz",
            log_normalized=log_expression[indices],
            coordinates_xy=coordinates,
            genes=np.asarray(genes),
            barcodes=frame["barcode_original"].astype(str).to_numpy(),
        )
        print(f"  Section_{number} <- {sample}: {len(records)} records")

    audit["combined"] = {
        "spots": int(merged.n_obs),
        "shared_genes_before_hvg": int(sections[0].n_vars),
        "selected_hvg": len(genes),
        "pca_components": 50,
        "pca_explained_variance_ratio_sum": float(np.sum(merged.uns["pca"]["variance_ratio"])),
        "gene_list_sha256": hashlib.sha256("\n".join(genes).encode()).hexdigest(),
        "note": "author preprocessing code is unavailable; this is a Methods-based reconstruction",
    }
    (target / "preprocessing_manifest.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, samples in DATASETS.items():
        process_dataset(name, samples)
    print(f"完成：{OUT}")


if __name__ == "__main__":
    main()
