"""在 HEST-IDC 官方四折协议上运行 ImageNet ResNet18/50 + Ridge。

ResNet50 路径复现 HEST 的公开基线：取 timm ResNet50 layer3（1,024 维）、
PCA-256、无截距 Ridge；ResNet18 用同一评测协议，检验本项目共享轻量编码器
在外部乳腺癌组织上的表现。脚本无命令行参数，特征会缓存后重复使用。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import scanpy as sc
import timm
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torchvision import models


sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw/hest_bench/IDC"
CACHE = ROOT / "data/processed/hest_idc/features"
RESULTS = ROOT / "results/hest_idc"
SAMPLES = ("NCBI783", "NCBI785", "TENX95", "TENX99")
ENCODERS = ("resnet18_shared", "resnet50_hest")


class ResNet50HEST(torch.nn.Module):
    """HEST/TRIDENT 的 ResNet50：layer3 feature map + global average pool。"""

    def __init__(self):
        super().__init__()
        self.features = timm.create_model(
            "resnet50", pretrained=False, features_only=True, out_indices=[3], num_classes=0
        )
        # timm 的 ResNet 参数名与 torchvision 相同；复用官方 ImageNet V1 权重，
        # 避免受 Hugging Face 模型下载连接影响。
        state = models.ResNet50_Weights.IMAGENET1K_V1.get_state_dict(progress=True)
        missing, unexpected = self.features.load_state_dict(state, strict=False)
        # out_indices=[3] 会在 timm 包装器中裁掉 layer4 与分类头；这些未使用的
        # ImageNet 权重因此应当出现在 unexpected 中，真正使用的 layer1-3 不能缺失。
        allowed_unused = all(key.startswith(("layer4.", "fc.")) for key in unexpected)
        if missing or not allowed_unused:
            raise RuntimeError(f"ResNet50 权重不匹配：missing={missing}, unexpected={unexpected}")

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(self.features(image)[0], 1).flatten(1)


def make_encoder(name: str) -> torch.nn.Module:
    if name == "resnet50_hest":
        return ResNet50HEST()
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    return torch.nn.Sequential(*list(model.children())[:-1], torch.nn.Flatten())


def read_barcodes(handle: h5py.File) -> np.ndarray:
    key = "barcodes" if "barcodes" in handle else "barcode"
    raw = handle[key][:].reshape(-1)
    return np.asarray([item.decode() if isinstance(item, bytes) else str(item) for item in raw])


def extract_features(encoder_name: str, sample: str, device: torch.device) -> Path:
    target_dir = CACHE / encoder_name
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{sample}.npz"
    if target.exists():
        return target
    model = make_encoder(encoder_name).eval().to(device)
    source = DATA / "patches" / f"{sample}.h5"
    features = []
    with h5py.File(source, "r") as handle:
        image_key = "img" if "img" in handle else "imgs"
        barcodes = read_barcodes(handle)
        for start in range(0, len(barcodes), 128):
            images = np.asarray(handle[image_key][start : start + 128])
            tensor = torch.from_numpy(images).permute(0, 3, 1, 2).float().div_(255).to(device)
            # TRIDENT target size为224；HEST patch本身已是224，故只需ImageNet标准化。
            mean = torch.tensor((0.485, 0.456, 0.406), device=device)[None, :, None, None]
            std = torch.tensor((0.229, 0.224, 0.225), device=device)[None, :, None, None]
            tensor = (tensor - mean) / std
            with torch.inference_mode(), torch.amp.autocast("cuda", enabled=device.type == "cuda", dtype=torch.float16):
                features.append(model(tensor).float().cpu().numpy())
            print(f"\r{encoder_name}/{sample}: {min(start + 128, len(barcodes))}/{len(barcodes)}", end="")
    print()
    matrix = np.concatenate(features).astype(np.float32)
    np.savez_compressed(target, features=matrix, barcodes=barcodes)
    del model
    torch.cuda.empty_cache()
    return target


def load_sample(encoder: str, sample: str, genes: list[str]) -> tuple[np.ndarray, np.ndarray]:
    with np.load(CACHE / encoder / f"{sample}.npz") as source:
        features = source["features"].astype(np.float32)
        barcodes = source["barcodes"].astype(str).tolist()
    adata = sc.read_h5ad(DATA / "adata" / f"{sample}.h5ad")
    expression = adata[barcodes, genes].copy()
    # 精确复用 HEST normalize_adata：输入 HEST h5ad 已规范化，此处只做 log1p。
    expression.X = expression.X.astype(np.float64)
    sc.pp.log1p(expression)
    return features, expression.to_df().to_numpy(dtype=np.float32)


def fold_metrics(true: np.ndarray, predicted: np.ndarray, genes: list[str]) -> dict:
    correlations = []
    for index, gene in enumerate(genes):
        value = float(pearsonr(true[:, index], predicted[:, index]).statistic)
        correlations.append({"gene": gene, "pcc": value})
    values = np.asarray([item["pcc"] for item in correlations])
    return {
        "pcc_macro": float(np.nanmean(values)),
        "pcc_std_across_genes": float(np.nanstd(values)),
        "mae": float(np.mean(np.abs(true - predicted))),
        "rmse": float(np.sqrt(np.mean((true - predicted) ** 2))),
        "per_gene": correlations,
    }


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    genes = json.loads((DATA / "var_50genes.json").read_text(encoding="utf-8"))["genes"]
    started = time.perf_counter()
    for encoder in ENCODERS:
        for sample in SAMPLES:
            extract_features(encoder, sample, device)

    all_results = {
        "protocol": {
            "dataset": "HEST-Benchmark IDC",
            "samples": list(SAMPLES),
            "folds": 4,
            "genes": "official var_50genes.json",
            "regression": "StandardScaler + PCA(256) + Ridge(alpha=100/(256*50), fit_intercept=False)",
            "official_current_resnet50_reference_pcc": 0.4739,
        },
        "encoders": {},
    }
    for encoder in ENCODERS:
        loaded = {sample: load_sample(encoder, sample, genes) for sample in SAMPLES}
        folds = []
        for fold in range(4):
            train_ids = __import__("pandas").read_csv(DATA / "splits" / f"train_{fold}.csv")["sample_id"].tolist()
            test_ids = __import__("pandas").read_csv(DATA / "splits" / f"test_{fold}.csv")["sample_id"].tolist()
            train_x = np.concatenate([loaded[sample][0] for sample in train_ids])
            train_y = np.concatenate([loaded[sample][1] for sample in train_ids])
            test_x = np.concatenate([loaded[sample][0] for sample in test_ids])
            test_y = np.concatenate([loaded[sample][1] for sample in test_ids])
            transform = Pipeline([
                ("scale", StandardScaler()),
                ("pca", PCA(n_components=256, random_state=1)),
            ])
            train_x = transform.fit_transform(train_x)
            test_x = transform.transform(test_x)
            ridge = Ridge(alpha=100 / (256 * 50), solver="lsqr", fit_intercept=False, max_iter=1000)
            prediction = ridge.fit(train_x, train_y).predict(test_x)
            metrics = fold_metrics(test_y, prediction, genes)
            folds.append({"fold": fold, "train": train_ids, "test": test_ids, "spots_train": len(train_y), "spots_test": len(test_y), "metrics": metrics})
            print(f"{encoder} fold={fold} test={test_ids} PCC={metrics['pcc_macro']:.4f}")
        values = [fold["metrics"]["pcc_macro"] for fold in folds]
        all_results["encoders"][encoder] = {
            "folds": folds,
            "pcc_mean_across_folds": float(np.mean(values)),
            "pcc_std_across_folds": float(np.std(values)),
        }
    all_results["runtime_seconds"] = time.perf_counter() - started
    (RESULTS / "evaluation.json").write_text(json.dumps(all_results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"完成：{RESULTS / 'evaluation.json'}")


if __name__ == "__main__":
    main()
