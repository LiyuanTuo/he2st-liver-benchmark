"""把 GEO 下载的 .tiff.gz 流式解压为 data/processed/gse240429/tiff/*.tif。

原因：tifffile 的内存映射读取需要一个可直接 seek 的未压缩文件；
全分辨率 WSI 很大，训练裁图时反复解压 gzip 太慢。原始 .gz 仍保留在
data/raw 中作为来源凭证。

本脚本幂等：目标 .tif 已存在且大小非零时跳过。
"""

from __future__ import annotations

import gzip
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/GSE240429"
OUTPUT = ROOT / "data/processed/gse240429/tiff"

SLIDES = ("C73_A1", "C73_B1", "C73_C1", "C73_D1")


def gz_name(slide: str) -> str:
    gsm_number = 7697868 + SLIDES.index(slide)  # A1..D1 -> GSM7697868..71
    return f"GSM{gsm_number}_GEX_{slide}_Merged.tiff.gz"


def gz_uncompressed_size(path: Path) -> int:
    """读取 gzip 尾部 ISIZE（解压后大小 mod 2^32），用于幂等校验。"""

    with open(path, "rb") as source:
        source.seek(-4, 2)
        return int.from_bytes(source.read(4), "little")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for slide in SLIDES:
        source = RAW / gz_name(slide)
        target = OUTPUT / f"{slide}.tif"
        if not source.exists():
            print(f"缺失来源文件：{source}，请先补下")
            continue
        expected = gz_uncompressed_size(source)
        if target.exists() and target.stat().st_size == expected:
            print(f"跳过 {slide}（已存在且大小匹配 {expected / 1e9:.2f} GB）")
            continue
        print(f"解压 {slide}: {source.name} ...")
        with gzip.open(source, "rb") as src, open(target, "wb") as dst:
            shutil.copyfileobj(src, dst, length=64 * 1024 * 1024)
        if target.stat().st_size != expected:
            raise RuntimeError(
                f"{target} 解压后大小 {target.stat().st_size} 与 gzip ISIZE "
                f"{expected} 不一致，请检查来源文件是否完整"
            )
        print(f"  -> {target} ({target.stat().st_size / 1e9:.2f} GB)")
    print("完成")


if __name__ == "__main__":
    main()
