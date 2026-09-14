"""把真实 H&E、Visium spot 和实测基因图放在同一张说明图中。

使用 Space Ranger 的 hires PNG，因此适合任务入门和配准检查；模型训练使用的
224×224 全分辨率 patch 会在完整 TIFF 解压后另行生成。
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/GSE240429"
ARRAY = ROOT / "data/processed/gse240429/arrays/C73_A1.npz"
OUTPUT = ROOT / "figures/03_he_and_st_task_example.png"


def main() -> None:
    image = np.asarray(Image.open(RAW / "GSM7697868_C73A1_tissue_hires_image.png"))
    scale = json.loads(
        (RAW / "GSM7697868_C73A1_scalefactors_json.json").read_text(encoding="utf-8")
    )["tissue_hires_scalef"]

    with np.load(ARRAY) as data:
        coordinates = data["coordinates_xy"] * scale
        expression = data["log_normalized"]
        genes = data["genes"].astype(str).tolist()
    glul = expression[:, genes.index("GLUL")]

    figure, axes = plt.subplots(1, 4, figsize=(18, 5), constrained_layout=True)
    axes[0].imshow(image)
    axes[0].set_title("A. Real H&E histology\n(human liver, C73 A1)")

    axes[1].imshow(image)
    axes[1].scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        s=7,
        facecolors="none",
        edgecolors="#00E5FF",
        linewidths=0.35,
        alpha=0.75,
    )
    axes[1].set_title("B. Registered measurement spots\n2,378 matched locations")

    # 放大一个高 GLUL 区域，仅用于展示蓝紫色细胞核和粉红色组织结构。
    center = coordinates[int(np.argmax(glul))]
    half_width = 170
    left, right = int(center[0] - half_width), int(center[0] + half_width)
    top, bottom = int(center[1] - half_width), int(center[1] + half_width)
    axes[2].imshow(image[max(top, 0) : bottom, max(left, 0) : right])
    axes[2].set_title("C. H&E morphology detail\n(nuclei and tissue architecture)")

    axes[3].imshow(image)
    points = axes[3].scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        c=glul,
        s=9,
        cmap="magma",
        linewidths=0,
        alpha=0.85,
    )
    figure.colorbar(points, ax=axes[3], shrink=0.68, label="measured log-normalized GLUL")
    axes[3].set_title("D. One column of the ST matrix\n(GLUL plotted at each spot)")

    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(
        "H&E-to-ST prediction: learn morphology/position → expression vector at every spot",
        fontsize=15,
    )
    figure.savefig(OUTPUT, dpi=220)
    plt.close(figure)
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
