"""构造 ResSAT 统一协议数据集（GSE240429）。

按官方数据格式生成 data/processed/ressat_gse240429/：
    Section_{i}/
        dataset.pkl    list of (patch [3,224,224] float32, pca-50 target)
        locations.pkl  list of normalized (x,y) in [0,1]
    pca_info.pkl       {'mean','std','components','gene_names'}
    gene_list.csv      200 个基因（列名 gene）

关键差异与理由（统一协议，非官方预处理复现）：
- 官方用 2,000 HVG + Harmony 校正；本项目统一协议用预先固定的 200 基因
  （SHA256 已记录），与 BLEEP/ST-Net/Stem/GenAR 共用同一面板与划分；
- PCA 目标只用训练切片（Section_3=A1, Section_4=B1）拟合 50 维，再
  transform 到验证/测试切片，避免测试分布泄漏；
- patch 直接复用 scripts/05 缓存的 224×224 图像块，保证与其它方法输入一致。

切片到 Section 的映射沿用官方 load_sections 语义：
    Section_1 = 测试 C73_D1，Section_2 = 验证 C73_C1，
    Section_3/4 = 训练 C73_A1 / C73_B1。
"""

from __future__ import annotations

import csv
import pickle
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
ARRAYS = ROOT / "data/processed/gse240429/arrays"
PATCHES = ROOT / "data/processed/gse240429/image"
OUTPUT = ROOT / "data/processed/ressat_gse240429"

# (section_dir, slide, role)
SECTIONS = [
    ("Section_1", "C73_D1", "test"),
    ("Section_2", "C73_C1", "validation"),
    ("Section_3", "C73_A1", "train"),
    ("Section_4", "C73_B1", "train"),
]
PCA_DIM = 50


def load_slide(slide: str) -> dict[str, np.ndarray]:
    with np.load(ARRAYS / f"{slide}.npz") as source:
        data = {name: source[name] for name in source.files}
    patches = np.load(PATCHES / f"{slide}_patches_uint8.npy", mmap_mode="r")
    if len(patches) != len(data["log_normalized"]):
        raise ValueError(f"{slide} patch 行数与表达行数不一致")
    return data


def normalize_coordinates(coordinates: np.ndarray) -> np.ndarray:
    low = coordinates.min(axis=0)
    span = coordinates.max(axis=0) - low
    span[span == 0] = 1.0
    return (coordinates - low) / span


def main() -> None:
    slides = {slide: load_slide(slide) for _, slide, _ in SECTIONS}
    train_names = [slide for _, slide, role in SECTIONS if role == "train"]

    # 只用训练切片拟合 PCA：200 基因 log-normalized 表达 -> 50 维。
    train_matrix = np.concatenate(
        [slides[name]["log_normalized"] for name in train_names], axis=0
    ).astype(np.float64)
    mean = train_matrix.mean(axis=0)
    std = train_matrix.std(axis=0)
    std[std < 1e-8] = 1.0
    standardized = (train_matrix - mean) / std
    pca = PCA(n_components=PCA_DIM, random_state=42)
    pca.fit(standardized)

    genes = [str(gene) for gene in slides[train_names[0]]["genes"]]
    pca_info = {
        "mean": mean,
        "std": std,
        "components": pca.components_,
        "gene_names": genes,
    }

    for section_dir, slide, _role in SECTIONS:
        section_path = OUTPUT / section_dir
        section_path.mkdir(parents=True, exist_ok=True)
        data = slides[slide]
        matrix = data["log_normalized"].astype(np.float64)
        scores = pca.transform((matrix - mean) / std).astype(np.float32)
        patches = np.load(PATCHES / f"{slide}_patches_uint8.npy", mmap_mode="r")
        coordinates = normalize_coordinates(data["coordinates_xy"].astype(np.float64))

        # 官方 dataset.pkl：list of (patch, target)。这里存 uint8 HWC（0-255），
        # 因为现代 torchvision 的 ToPILImage 会把 float32 数组按 [0,1] 缩放到
        # [0,255]（0-255 的 float patch 会数值回绕损坏）；uint8 则原样通过。
        records = []
        for index in range(len(scores)):
            patch = np.ascontiguousarray(patches[index]).astype(np.uint8)
            records.append((patch, scores[index]))
        with open(section_path / "dataset.pkl", "wb") as target:
            pickle.dump(records, target, protocol=4)

        locations = [tuple(round(float(value), 6) for value in row) for row in coordinates]
        with open(section_path / "locations.pkl", "wb") as target:
            pickle.dump(locations, target, protocol=4)
        print(f"{section_dir} <- {slide}: {len(records)} spots, "
              f"patch {patches.shape[1:]} -> {patch.shape}")

    with open(OUTPUT / "pca_info.pkl", "wb") as target:
        pickle.dump(pca_info, target, protocol=4)
    with open(OUTPUT / "gene_list.csv", "w", newline="", encoding="utf-8") as target:
        writer = csv.writer(target)
        writer.writerow(["gene"])
        for gene in genes:
            writer.writerow([gene])

    print(f"PCA fitted on {len(train_matrix)} train spots, {len(genes)} genes -> "
          f"{PCA_DIM} dims; explained variance "
          f"{float(pca.explained_variance_ratio_.sum()):.3f}")
    print(f"完成：{OUTPUT}")


if __name__ == "__main__":
    main()
