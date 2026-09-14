"""下载 ResSAT 论文实际使用的四个 10x Visium 小鼠脑切片。

脚本不接收命令行参数，直接运行即可。它只下载复现实验需要的三类文件：
表达矩阵、Space Ranger 空间坐标包和原始 H&E TIFF。中断后再次运行会续传。
"""

from __future__ import annotations

import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

# Windows 终端偶尔继承 cp1252；统一成 UTF-8，确保中文进度信息可读。
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "ressat_mouse_brain"
BASE = "https://cf.10xgenomics.com/samples/spatial-exp"

# 名称和 Space Ranger 版本均来自 ResSAT 论文引用的 10x 官方数据页。
SAMPLES = {
    "SA1": ("1.1.0", "V1_Mouse_Brain_Sagittal_Anterior"),
    "SA2": ("1.1.0", "V1_Mouse_Brain_Sagittal_Anterior_Section_2"),
    "SP1": ("1.0.0", "V1_Mouse_Brain_Sagittal_Posterior"),
    "SP2": ("1.0.0", "V1_Mouse_Brain_Sagittal_Posterior_Section_2"),
}
SUFFIXES = ("filtered_feature_bc_matrix.h5", "spatial.tar.gz", "image.tif")


def sha256(path: Path) -> str:
    """分块计算文件摘要，避免把数百 MB 的 TIFF 一次读入内存。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, path: Path) -> None:
    """下载单个文件；已有的 .part 文件会从断点继续。"""
    partial = path.with_suffix(path.suffix + ".part")
    existing = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}
    with requests.get(url, headers=headers, stream=True, timeout=(30, 300)) as response:
        response.raise_for_status()
        # 服务器若忽略 Range 并返回 200，就重新写入，避免得到重复拼接文件。
        mode = "ab" if existing and response.status_code == 206 else "wb"
        total = int(response.headers.get("content-length", 0)) + (existing if mode == "ab" else 0)
        done = existing if mode == "ab" else 0
        with partial.open(mode) as handle:
            for block in response.iter_content(8 * 1024 * 1024):
                if not block:
                    continue
                handle.write(block)
                done += len(block)
                # 每 64 MiB 报告一次；并行下载时不会用回车互相覆盖。
                if done == total or done // 2**26 != (done - len(block)) // 2**26:
                    print(f"  {path.parent.name}/{path.name}: {done / 2**20:.1f}/{total / 2**20:.1f} MiB")
    partial.replace(path)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {"source": "10x Genomics", "samples": {}}
    jobs: list[tuple[str, Path]] = []
    for short_name, (version, sample_name) in SAMPLES.items():
        sample_dir = OUT / short_name
        sample_dir.mkdir(exist_ok=True)
        for suffix in SUFFIXES:
            filename = f"{sample_name}_{suffix}"
            url = f"{BASE}/{version}/{sample_name}/{filename}"
            path = sample_dir / filename
            if not path.exists():
                jobs.append((url, path))
            else:
                print(f"已存在：{short_name}/{path.name}")

    # 四路并行能显著减轻 10x 单连接限速；每个文件本身仍支持断点续传。
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(download, url, path): path for url, path in jobs}
        for future in as_completed(futures):
            future.result()
            print(f"下载完成：{futures[future].parent.name}/{futures[future].name}")

    # 全部下载结束后再生成可审计清单和 SHA-256。
    for short_name, (version, sample_name) in SAMPLES.items():
        files = []
        sample_dir = OUT / short_name
        for suffix in SUFFIXES:
            filename = f"{sample_name}_{suffix}"
            url = f"{BASE}/{version}/{sample_name}/{filename}"
            path = sample_dir / filename
            files.append({"name": filename, "bytes": path.stat().st_size, "sha256": sha256(path), "url": url})
        manifest["samples"][short_name] = {"version": version, "sample_name": sample_name, "files": files}
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"完成，校验清单：{OUT / 'manifest.json'}")


if __name__ == "__main__":
    main()
