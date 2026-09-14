"""下载 HEST-Benchmark 的 IDC 单任务（4 张切片），不下载 42 GB 全仓库。

保留运行官方四折 benchmark 所需的 patch HDF5、表达 h5ad、固定划分和
50 基因列表。脚本无参数、四路并行、支持断点续传并输出 SHA-256 清单。
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/raw/hest_bench/IDC"
BASE = "https://huggingface.co/datasets/MahmoodLab/hest-bench/resolve/main/IDC"
SAMPLES = ("NCBI783", "NCBI785", "TENX95", "TENX99")
FILES = (
    [f"patches/{sample}.h5" for sample in SAMPLES]
    + [f"adata/{sample}.h5ad" for sample in SAMPLES]
    + [f"splits/{kind}_{fold}.csv" for fold in range(4) for kind in ("train", "test")]
    + ["mean_50genes.json", "var_50genes.json"]
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(relative: str) -> Path:
    path = OUT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{BASE}/{relative}?download=true"
    # 先解析 Hugging Face/Xet 重定向；某些网络环境下直接流式 GET 会停住，
    # 但最终 CDN 地址稳定可用。
    head = requests.head(url, allow_redirects=True, timeout=(30, 60))
    head.raise_for_status()
    expected = int(head.headers.get("content-length", 0))
    if path.exists() and (not expected or path.stat().st_size == expected):
        return path
    partial = path.with_suffix(path.suffix + ".part")
    curl = shutil.which("curl.exe") or shutil.which("curl")
    if curl is None:
        raise RuntimeError("需要系统 curl 执行可续传的 Xet/CDN 下载")
    command = [curl, "-4", "--fail", "--location", "--retry", "10", "--retry-all-errors", "--connect-timeout", "20", "--silent", "--show-error"]
    if partial.exists() and partial.stat().st_size:
        command += ["-C", "-"]
    command += ["-o", str(partial), head.url]
    subprocess.run(command, check=True)
    if expected and partial.stat().st_size != expected:
        raise IOError(f"{relative} 下载尺寸错误：{partial.stat().st_size} != {expected}")
    partial.replace(path)
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(download, relative): relative for relative in FILES}
        for future in as_completed(futures):
            future.result()
            print(f"完成：{futures[future]}")
    manifest = {
        "source": "MahmoodLab/hest-bench (Hugging Face)",
        "task": "IDC",
        "files": [
            {"path": relative, "bytes": (OUT / relative).stat().st_size, "sha256": sha256(OUT / relative)}
            for relative in FILES
        ],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"校验清单：{OUT / 'manifest.json'}")


if __name__ == "__main__":
    main()
