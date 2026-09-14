"""在统一切片划分上训练三种 H&E 图像方法。

方法：
- Image Ridge：冻结 ResNet18 特征后的线性回归；
- ST-Net style MLP：冻结图像特征后的非线性多基因回归；
- BLEEP：直接复用官方 ProjectionHead 和软标签对比损失，再以训练 spot
  的表达嵌入做近邻插补。

三者都只使用训练切片拟合；验证切片用于选超参数/早停；测试切片只在
最后评测一次。此处是统一数据协议下的适配实验，不冒充各论文原表格数值。
"""

from __future__ import annotations

import json
import random
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party/BLEEP"))

from he2st.metrics import evaluate_normalized
from modules import ProjectionHead  # BLEEP 官方实现


ARRAYS = ROOT / "data/processed/gse240429/arrays"
IMAGES = ROOT / "data/processed/gse240429/image"
OUTPUT = ROOT / "results/unified_image_baselines"
TRAIN_SLIDES = ("C73_A1", "C73_B1")
VALIDATION_SLIDE = "C73_C1"
TEST_SLIDE = "C73_D1"


def load_slide(slide: str) -> dict[str, np.ndarray]:
    """读取图像特征和同顺序表达；预处理脚本已用 barcode 哈希核对顺序。"""

    with np.load(ARRAYS / f"{slide}.npz") as source:
        data = {name: source[name] for name in source.files}
    data["features"] = np.load(IMAGES / f"{slide}_resnet18.npy").copy()
    if len(data["features"]) != len(data["log_normalized"]):
        raise ValueError(f"{slide} 图像特征与表达行数不一致")
    return data


def json_ready(value):
    """把 NumPy 类型和 NaN 转成标准 JSON 可保存的值。"""

    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def evaluate(true: np.ndarray, predicted: np.ndarray, coordinates: np.ndarray) -> dict:
    return evaluate_normalized(true, predicted, coordinates)[0]


class ExpressionMLP(nn.Module):
    """ST-Net 风格的轻量表达回归头。"""

    def __init__(self, gene_count: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(256, gene_count),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class FrozenBLEEP(nn.Module):
    """在共同的冻结图像特征上使用 BLEEP 官方双投影头。"""

    def __init__(self, gene_count: int):
        super().__init__()
        self.image_projection = ProjectionHead(512, projection_dim=256, dropout=0.1)
        self.spot_projection = ProjectionHead(gene_count, projection_dim=256, dropout=0.1)

    def embeddings(self, image: torch.Tensor, expression: torch.Tensor):
        return self.image_projection(image), self.spot_projection(expression)

    def forward(self, image: torch.Tensor, expression: torch.Tensor) -> torch.Tensor:
        image_embedding, spot_embedding = self.embeddings(image, expression)
        logits = spot_embedding @ image_embedding.T
        image_similarity = image_embedding @ image_embedding.T
        spot_similarity = spot_embedding @ spot_embedding.T
        targets = F.softmax((image_similarity + spot_similarity) / 2.0, dim=-1)
        spot_loss = -(targets * F.log_softmax(logits, dim=-1)).sum(dim=-1)
        image_loss = -(targets.T * F.log_softmax(logits.T, dim=-1)).sum(dim=-1)
        return ((spot_loss + image_loss) / 2.0).mean()


def train_with_early_stopping(
    model: nn.Module,
    train_loader: DataLoader,
    validation_tensors: tuple[torch.Tensor, torch.Tensor],
    loss_function,
    epochs: int = 150,
    patience: int = 20,
) -> tuple[nn.Module, dict[str, object]]:
    """按验证损失保存最佳权重，避免使用测试标签选 epoch。"""

    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_state, best_loss, stale = None, float("inf"), 0
    history = []
    validation_x, validation_y = (tensor.to(device) for tensor in validation_tensors)

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model, x, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        model.eval()
        with torch.inference_mode():
            validation_loss = float(loss_function(model, validation_x, validation_y).item())
        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "validation_loss": validation_loss,
        })
        if validation_loss < best_loss - 1e-6:
            best_loss, stale = validation_loss, 0
            best_state = deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= patience:
                break
        if epoch == 1 or epoch % 10 == 0:
            print(f"  epoch={epoch:3d} train={np.mean(losses):.5f} val={validation_loss:.5f}")

    assert best_state is not None
    model.load_state_dict(best_state)
    return model, {"best_validation_loss": best_loss, "epochs_run": len(history), "history": history}


@torch.inference_mode()
def predict_mlp(model: nn.Module, features: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    result = []
    for start in range(0, len(features), 512):
        batch = torch.from_numpy(features[start : start + 512]).float().to(device)
        result.append(model(batch).cpu().numpy())
    return np.concatenate(result)


@torch.inference_mode()
def bleeper_embeddings(
    model: FrozenBLEEP,
    features: np.ndarray,
    expression: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """分批获得图像查询向量和表达参考向量。"""

    model.eval()
    image_result, spot_result = [], []
    for start in range(0, len(features), 512):
        stop = start + 512
        x = torch.from_numpy(features[start:stop]).float().to(device)
        y = torch.from_numpy(expression[start:stop]).float().to(device)
        image, spot = model.embeddings(x, y)
        image_result.append(F.normalize(image, dim=-1).cpu().numpy())
        spot_result.append(F.normalize(spot, dim=-1).cpu().numpy())
    return np.concatenate(image_result), np.concatenate(spot_result)


def retrieve_expression(
    query_image_embedding: np.ndarray,
    reference_spot_embedding: np.ndarray,
    reference_expression: np.ndarray,
    neighbors: int,
) -> np.ndarray:
    """复现 BLEEP 推理：图像查询与训练表达嵌入匹配，再平均其表达。"""

    similarities = query_image_embedding @ reference_spot_embedding.T
    indices = np.argpartition(similarities, -neighbors, axis=1)[:, -neighbors:]
    return reference_expression[indices].mean(axis=1)


def main() -> None:
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_parts = [load_slide(slide) for slide in TRAIN_SLIDES]
    validation = load_slide(VALIDATION_SLIDE)
    test = load_slide(TEST_SLIDE)
    train_x = np.concatenate([part["features"] for part in train_parts]).astype(np.float32)
    train_y = np.concatenate([part["log_normalized"] for part in train_parts]).astype(np.float32)
    validation_x = validation["features"].astype(np.float32)
    validation_y = validation["log_normalized"].astype(np.float32)
    test_x = test["features"].astype(np.float32)

    # 只用训练切片估计特征标准化参数。
    feature_mean = train_x.mean(axis=0, keepdims=True)
    feature_std = np.maximum(train_x.std(axis=0, keepdims=True), 1e-6)
    train_x = (train_x - feature_mean) / feature_std
    validation_x = (validation_x - feature_mean) / feature_std
    test_x = (test_x - feature_mean) / feature_std

    summary: dict[str, object] = {
        "protocol": {
            "train": list(TRAIN_SLIDES),
            "validation": VALIDATION_SLIDE,
            "test": TEST_SLIDE,
            "genes": train_y.shape[1],
            "image_encoder": "frozen torchvision ResNet18 IMAGENET1K_V1",
        },
        "validation_selection": {},
        "test_metrics": {},
        "runtime_seconds": {},
    }

    # 1) 线性图像基线。
    started = time.perf_counter()
    ridge_scores = {}
    for alpha in (1.0, 10.0, 100.0, 1000.0):
        model = Ridge(alpha=alpha).fit(train_x, train_y)
        ridge_scores[str(alpha)] = evaluate(
            validation_y, model.predict(validation_x), validation["coordinates_xy"]
        )["pcc_macro_all_fixed_genes"]
    best_alpha = float(max(ridge_scores, key=ridge_scores.get))
    ridge = Ridge(alpha=best_alpha).fit(train_x, train_y)
    predictions = {"image_ridge": ridge.predict(test_x).astype(np.float32)}
    summary["validation_selection"]["image_ridge"] = {
        "pcc_by_alpha": ridge_scores,
        "selected_alpha": best_alpha,
    }
    summary["runtime_seconds"]["image_ridge"] = time.perf_counter() - started

    tensors = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    loader = DataLoader(tensors, batch_size=128, shuffle=True, num_workers=0)
    validation_tensors = (torch.from_numpy(validation_x), torch.from_numpy(validation_y))

    # 2) ST-Net 风格 MLP。
    started = time.perf_counter()
    mlp = ExpressionMLP(train_y.shape[1]).to(device)
    mlp, mlp_info = train_with_early_stopping(
        mlp,
        loader,
        validation_tensors,
        lambda model, x, y: F.mse_loss(model(x), y),
    )
    predictions["stnet_style_mlp"] = predict_mlp(mlp, test_x, device)
    torch.save(mlp.state_dict(), OUTPUT / "stnet_style_mlp.pt")
    summary["validation_selection"]["stnet_style_mlp"] = mlp_info
    summary["runtime_seconds"]["stnet_style_mlp"] = time.perf_counter() - started

    # 3) BLEEP 双模态对比学习与近邻表达插补。
    started = time.perf_counter()
    bleep = FrozenBLEEP(train_y.shape[1]).to(device)
    bleep, bleep_info = train_with_early_stopping(
        bleep,
        loader,
        validation_tensors,
        lambda model, x, y: model(x, y),
    )
    train_image_embedding, train_spot_embedding = bleeper_embeddings(
        bleep, train_x, train_y, device
    )
    validation_image_embedding, _ = bleeper_embeddings(
        bleep, validation_x, validation_y, device
    )
    test_image_embedding, _ = bleeper_embeddings(
        bleep, test_x, test["log_normalized"].astype(np.float32), device
    )
    neighbor_scores = {}
    for neighbors in (1, 5, 10, 20, 50):
        validation_prediction = retrieve_expression(
            validation_image_embedding, train_spot_embedding, train_y, neighbors
        )
        neighbor_scores[str(neighbors)] = evaluate(
            validation_y, validation_prediction, validation["coordinates_xy"]
        )["pcc_macro_all_fixed_genes"]
    best_neighbors = int(max(neighbor_scores, key=neighbor_scores.get))
    predictions["bleep"] = retrieve_expression(
        test_image_embedding, train_spot_embedding, train_y, best_neighbors
    ).astype(np.float32)
    torch.save(bleep.state_dict(), OUTPUT / "bleep.pt")
    summary["validation_selection"]["bleep"] = {
        **bleep_info,
        "retrieval_pcc_by_neighbors": neighbor_scores,
        "selected_neighbors": best_neighbors,
    }
    summary["runtime_seconds"]["bleep"] = time.perf_counter() - started

    for name, predicted in predictions.items():
        metrics, per_gene_pcc = evaluate_normalized(
            test["log_normalized"], predicted, test["coordinates_xy"]
        )
        summary["test_metrics"][name] = metrics
        np.savez_compressed(
            OUTPUT / f"{name}_C73_D1.npz",
            predicted=predicted,
            true=test["log_normalized"],
            coordinates_xy=test["coordinates_xy"],
            genes=test["genes"],
            per_gene_pcc=per_gene_pcc,
        )
    (OUTPUT / "metrics.json").write_text(
        json.dumps(json_ready(summary), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(json_ready(summary["test_metrics"]), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
