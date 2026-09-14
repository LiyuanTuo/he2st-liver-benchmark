"""训练不看 H&E 的基线，量化“仅靠均值/坐标能做到多少”。

这些模型不是候选最终方法，而是必要的对照：如果复杂模型没有明显超过它们，
就不能把性能归因于学习到了病理形态。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from he2st.metrics import evaluate_normalized

DATA = ROOT / "data/processed/gse240429/arrays"
RESULTS = ROOT / "results/nonimage_baselines"


def json_ready(value):
    """把未定义指标的 NaN 转成标准 JSON 的 null。"""

    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def load(slide: str) -> dict[str, np.ndarray]:
    with np.load(DATA / f"{slide}.npz") as source:
        return {name: source[name] for name in source.files}


def coordinate_features(coordinates: np.ndarray) -> np.ndarray:
    """每张切片独立归一化后，生成低阶与 Fourier 空间特征。"""

    coordinates = coordinates.astype(np.float64)
    minimum = coordinates.min(axis=0, keepdims=True)
    span = coordinates.max(axis=0, keepdims=True) - minimum
    normalized = (coordinates - minimum) / np.maximum(span, 1.0)
    x, y = normalized[:, 0:1], normalized[:, 1:2]
    columns = [x, y, x * y, x**2, y**2]
    for frequency in (1, 2, 4, 8):
        columns.extend(
            [
                np.sin(2 * np.pi * frequency * x),
                np.cos(2 * np.pi * frequency * x),
                np.sin(2 * np.pi * frequency * y),
                np.cos(2 * np.pi * frequency * y),
            ]
        )
    return np.concatenate(columns, axis=1)


def score(true: np.ndarray, predicted: np.ndarray, coordinates: np.ndarray) -> float:
    return float(evaluate_normalized(true, predicted, coordinates)[0]["pcc_macro_all_fixed_genes"])


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    train_slides = [load("C73_A1"), load("C73_B1")]
    validation = load("C73_C1")
    test = load("C73_D1")

    train_x = np.concatenate(
        [coordinate_features(slide["coordinates_xy"]) for slide in train_slides]
    )
    train_y = np.concatenate([slide["log_normalized"] for slide in train_slides])
    validation_x = coordinate_features(validation["coordinates_xy"])
    test_x = coordinate_features(test["coordinates_xy"])

    predictions: dict[str, np.ndarray] = {
        "train_gene_mean": np.broadcast_to(train_y.mean(axis=0), test["log_normalized"].shape),
    }
    tuning: dict[str, object] = {}

    ridge_candidates = [0.01, 0.1, 1.0, 10.0, 100.0]
    ridge_scores = {}
    for alpha in ridge_candidates:
        model = Ridge(alpha=alpha).fit(train_x, train_y)
        ridge_scores[str(alpha)] = score(
            validation["log_normalized"], model.predict(validation_x), validation["coordinates_xy"]
        )
    best_alpha = max(ridge_scores, key=ridge_scores.get)
    ridge = Ridge(alpha=float(best_alpha)).fit(train_x, train_y)
    predictions["coordinate_ridge"] = ridge.predict(test_x)
    tuning["coordinate_ridge"] = {"validation_pcc_by_alpha": ridge_scores, "selected_alpha": float(best_alpha)}

    neighbor_candidates = [4, 8, 16, 32, 64]
    neighbor_scores = {}
    for neighbors in neighbor_candidates:
        model = KNeighborsRegressor(n_neighbors=neighbors, weights="distance", n_jobs=-1).fit(
            train_x[:, :2], train_y
        )
        neighbor_scores[str(neighbors)] = score(
            validation["log_normalized"],
            model.predict(validation_x[:, :2]),
            validation["coordinates_xy"],
        )
    best_neighbors = int(max(neighbor_scores, key=neighbor_scores.get))
    knn = KNeighborsRegressor(n_neighbors=best_neighbors, weights="distance", n_jobs=-1).fit(
        train_x[:, :2], train_y
    )
    predictions["coordinate_knn"] = knn.predict(test_x[:, :2])
    tuning["coordinate_knn"] = {
        "validation_pcc_by_neighbors": neighbor_scores,
        "selected_neighbors": best_neighbors,
    }

    summary: dict[str, object] = {
        "protocol": {
            "train": ["C73_A1", "C73_B1"],
            "validation": ["C73_C1"],
            "test": ["C73_D1"],
            "note": "No H&E pixels or image embeddings are used.",
        },
        "tuning": tuning,
        "test_metrics": {},
    }
    for name, predicted in predictions.items():
        metrics, gene_pcc = evaluate_normalized(
            test["log_normalized"], predicted, test["coordinates_xy"]
        )
        summary["test_metrics"][name] = metrics
        np.savez_compressed(
            RESULTS / f"{name}_C73_D1.npz",
            predicted=predicted.astype(np.float32),
            true=test["log_normalized"].astype(np.float32),
            coordinates_xy=test["coordinates_xy"],
            genes=test["genes"],
            per_gene_pcc=gene_pcc,
        )

    clean_summary = json_ready(summary)
    (RESULTS / "metrics.json").write_text(
        json.dumps(clean_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(clean_summary["test_metrics"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
