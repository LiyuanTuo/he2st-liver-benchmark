"""从完整 H&E 切片裁出 spot 图像块，并缓存统一的 ResNet18 特征。

为什么单独做这一步：
1. 全分辨率 TIFF 很大，训练时反复打开和裁图会浪费大量时间；
2. ST-Net、BLEEP、Stem、GenAR 必须读取完全相同的图像和 barcode 顺序；
3. 保存图像块后，仍可训练端到端 CNN；保存特征后，可快速训练轻量模型。

本脚本没有命令行参数。数据放在 README 指定位置后直接运行即可。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import tifffile
import torch
from numpy.lib.format import open_memmap
from torchvision.models import ResNet18_Weights, resnet18


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/GSE240429"
TIFF_DIR = ROOT / "data/processed/gse240429/tiff"
ARRAYS = ROOT / "data/processed/gse240429/arrays"
OUTPUT = ROOT / "data/processed/gse240429/image"

SLIDES = ("C73_A1", "C73_B1", "C73_C1", "C73_D1")
PATCH_SIZE = 224
FEATURE_BATCH_SIZE = 128


def stable_hash(values: np.ndarray) -> str:
    """记录有顺序的 barcode 哈希，用于证明图像和表达逐行对齐。"""

    text = "\n".join(values.astype(str).tolist()) + "\n"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def locate_tiff(slide: str) -> Path:
    """在解压缓存目录中查找该切片的唯一全分辨率 TIFF（由 00_decompress_tiffs.py 生成）。"""

    candidate = TIFF_DIR / f"{slide}.tif"
    if candidate.exists() and candidate.stat().st_size > 0:
        return candidate
    raise FileNotFoundError(
        f"未找到 {candidate}；请先运行 scripts/00_decompress_tiffs.py 并确认 "
        f"data/raw/GSE240429 中的 .tiff.gz 完整"
    )


def canonical_image(array: np.ndarray) -> np.ndarray:
    """把 TIFF 的不同通道布局统一成 [height, width, RGB] uint8。"""

    image = np.asarray(array)
    image = np.squeeze(image)
    if image.ndim == 3 and image.shape[0] in (3, 4):
        image = np.moveaxis(image, 0, -1)
    if image.ndim != 3 or image.shape[-1] not in (3, 4):
        raise ValueError(f"不支持的 TIFF 形状：{image.shape}")
    image = image[..., :3]
    if image.dtype == np.uint16:
        image = (image / 257).astype(np.uint8)
    elif image.dtype != np.uint8:
        maximum = float(np.nanmax(image))
        scale = 255.0 / maximum if maximum > 1 else 255.0
        image = np.clip(image * scale, 0, 255).astype(np.uint8)
    return image


def crop_patch(image: np.ndarray, x: float, y: float) -> np.ndarray:
    """以全分辨率 `(x,y)` 为中心裁图；越界部分用白色补齐。"""

    half = PATCH_SIZE // 2
    center_x, center_y = int(round(x)), int(round(y))
    left, top = center_x - half, center_y - half
    right, bottom = left + PATCH_SIZE, top + PATCH_SIZE
    patch = np.full((PATCH_SIZE, PATCH_SIZE, 3), 255, dtype=np.uint8)

    source_left, source_top = max(left, 0), max(top, 0)
    source_right, source_bottom = min(right, image.shape[1]), min(bottom, image.shape[0])
    if source_left >= source_right or source_top >= source_bottom:
        raise ValueError(f"spot ({x:.1f},{y:.1f}) 完全位于图像之外 {image.shape[:2]}")

    target_left, target_top = source_left - left, source_top - top
    target_right = target_left + source_right - source_left
    target_bottom = target_top + source_bottom - source_top
    patch[target_top:target_bottom, target_left:target_right] = image[
        source_top:source_bottom, source_left:source_right
    ]
    return patch


def extract_patches(slide: str, image: np.ndarray, coordinates: np.ndarray) -> Path:
    """按表达矩阵的行顺序裁图，并写成可内存映射的 `.npy`。"""

    output = OUTPUT / f"{slide}_patches_uint8.npy"
    patches = open_memmap(
        output,
        mode="w+",
        dtype=np.uint8,
        shape=(len(coordinates), PATCH_SIZE, PATCH_SIZE, 3),
    )
    for index, (x, y) in enumerate(coordinates):
        patches[index] = crop_patch(image, x, y)
        if (index + 1) % 500 == 0 or index + 1 == len(coordinates):
            print(f"  裁图 {index + 1}/{len(coordinates)}")
    patches.flush()
    return output


@torch.inference_mode()
def extract_resnet18_features(patch_path: Path) -> Path:
    """对缓存图像块提取 512 维 ImageNet ResNet18 全局特征。"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    patches = np.load(patch_path, mmap_mode="r")
    output = OUTPUT / patch_path.name.replace("_patches_uint8.npy", "_resnet18.npy")
    features = open_memmap(output, mode="w+", dtype=np.float32, shape=(len(patches), 512))
    mean = torch.tensor((0.485, 0.456, 0.406), device=device).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device=device).view(1, 3, 1, 1)

    for start in range(0, len(patches), FEATURE_BATCH_SIZE):
        stop = min(start + FEATURE_BATCH_SIZE, len(patches))
        # `.copy()` 使内存映射数组变成连续且可写的内存，避免 PyTorch 警告。
        batch = torch.from_numpy(np.asarray(patches[start:stop]).copy())
        batch = batch.permute(0, 3, 1, 2).to(device=device, dtype=torch.float32) / 255.0
        batch = (batch - mean) / std
        features[start:stop] = model(batch).cpu().numpy()
        print(f"  特征 {stop}/{len(patches)}")
    features.flush()
    return output


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "patch_size": PATCH_SIZE,
        "feature_encoder": "torchvision ResNet18 IMAGENET1K_V1; final fc removed",
        "coordinate_convention": "10x full-resolution (x,y); patch centered at coordinate",
        "slides": {},
    }

    for slide in SLIDES:
        print(f"\n{slide}")
        patch_path = OUTPUT / f"{slide}_patches_uint8.npy"
        feature_path = OUTPUT / f"{slide}_resnet18.npy"
        genar_path = (
            ROOT / "data/processed/gse240429/processed_data/spot_features_resnet18"
            / f"{slide}_resnet18.pt"
        )
        if (
            patch_path.exists()
            and feature_path.exists()
            and genar_path.exists()
        ):
            with np.load(ARRAYS / f"{slide}.npz") as data:
                expected_rows = len(data["barcodes"])
            if np.load(feature_path, mmap_mode="r").shape[0] == expected_rows:
                print(f"  跳过 {slide}（patch/特征/GenAR 文件已存在且行数一致）")
                continue
        with np.load(ARRAYS / f"{slide}.npz") as data:
            coordinates = data["coordinates_xy"].astype(np.float32)
            barcodes = data["barcodes"]
        tiff_path = locate_tiff(slide)
        print(f"  读取 {tiff_path.name}")
        image = canonical_image(tifffile.imread(tiff_path, out="memmap"))
        height, width = image.shape[:2]
        if coordinates[:, 0].max() >= width or coordinates[:, 1].max() >= height:
            raise ValueError(
                f"{slide} 坐标范围超过 TIFF：xy max={coordinates.max(0)}, image={(width, height)}"
            )
        patch_path = extract_patches(slide, image, coordinates)
        feature_path = extract_resnet18_features(patch_path)
        # GenAR 官方数据加载器规定了下面的目录和文件名；这里只做无损格式转换。
        genar_dir = ROOT / "data/processed/gse240429/processed_data/spot_features_resnet18"
        genar_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            torch.from_numpy(np.load(feature_path, mmap_mode="r").copy()),
            genar_dir / f"{slide}_resnet18.pt",
        )
        manifest["slides"][slide] = {
            "source_tiff": tiff_path.name,
            "source_tiff_bytes": tiff_path.stat().st_size,
            "image_width": width,
            "image_height": height,
            "spot_count": len(coordinates),
            "barcode_order_sha256": stable_hash(barcodes),
            "patch_file": patch_path.name,
            "feature_file": feature_path.name,
        }
        del image

    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"\n完成：{OUTPUT}")


if __name__ == "__main__":
    main()
