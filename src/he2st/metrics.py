"""H&E→空间表达任务的统一评测。

主指标始终是预先固定 200 基因上的宏平均 PCC，而不是测试后再挑最好
预测的基因。`paper_pcc_top*_test_selected` 仅为复现 GenAR/Stem 的论文口径，
名称明确指出它使用测试结果排序，不能代替主指标。
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata
from sklearn.neighbors import NearestNeighbors


def _validate(true: np.ndarray, predicted: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    true = np.asarray(true, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if true.ndim != 2 or true.shape != predicted.shape:
        raise ValueError(f"真值和预测必须是相同二维形状，得到 {true.shape} 与 {predicted.shape}")
    if not np.isfinite(true).all() or not np.isfinite(predicted).all():
        raise ValueError("真值或预测中存在 NaN/Inf")
    return true, predicted


def per_gene_pearson(true: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """逐基因计算跨 spot 的 PCC；常数列返回 NaN。"""

    true, predicted = _validate(true, predicted)
    if true.shape[0] == 0:
        return np.full(true.shape[1], np.nan, dtype=np.float64)
    true_centered = true - true.mean(axis=0, keepdims=True)
    predicted_centered = predicted - predicted.mean(axis=0, keepdims=True)
    denominator = np.sqrt(
        np.square(true_centered).sum(axis=0)
        * np.square(predicted_centered).sum(axis=0)
    )
    correlation = np.full(true.shape[1], np.nan, dtype=np.float64)
    # Repeated nonzero constants can leave roundoff after mean subtraction.
    valid = (denominator > 0) & (np.ptp(true, axis=0) > 1e-12) & (np.ptp(predicted, axis=0) > 1e-12)
    correlation[valid] = (
        true_centered[:, valid] * predicted_centered[:, valid]
    ).sum(axis=0) / denominator[valid]
    return correlation


def per_gene_morans_i(values: np.ndarray, coordinates: np.ndarray, neighbors: int = 6) -> np.ndarray:
    """用 spot 的 k 近邻图计算每个基因的 Moran's I。"""

    values = np.asarray(values, dtype=np.float64)
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if values.ndim != 2 or coordinates.shape != (values.shape[0], 2):
        raise ValueError("Moran's I 的表达与坐标形状不匹配")
    neighbor_count = min(neighbors + 1, len(coordinates))
    if len(coordinates) < 2:
        return np.full(values.shape[1], np.nan)
    # Passing X explicitly includes self; exclude by row ID, including tied
    # coordinates. kneighbors(X=None) already removes self and must not be sliced.
    queried = NearestNeighbors(n_neighbors=neighbor_count).fit(coordinates).kneighbors(
        coordinates, return_distance=False
    )
    indices = np.asarray([row[row != i][:neighbor_count - 1] for i, row in enumerate(queried)])
    centered = values - values.mean(axis=0, keepdims=True)
    denominator = np.square(centered).sum(axis=0)
    numerator = (centered[:, None, :] * centered[indices]).sum(axis=(0, 1))
    result = np.full(values.shape[1], np.nan)
    valid = denominator > 0
    # 二值有向 kNN 图的权重和 S0=n*k，因此 n/S0=1/k。
    result[valid] = numerator[valid] / (indices.shape[1] * denominator[valid])
    return result


def evaluate_normalized(
    true: np.ndarray,
    predicted: np.ndarray,
    coordinates: np.ndarray,
) -> tuple[dict[str, float | int], np.ndarray]:
    """评测 log-normalized 表达，并返回汇总指标和逐基因 PCC。"""

    true, predicted = _validate(true, predicted)
    pcc = per_gene_pearson(true, predicted)
    defined = pcc[np.isfinite(pcc)]
    absolute_error = np.abs(predicted - true)

    true_rank = rankdata(true, axis=0)
    predicted_rank = rankdata(predicted, axis=0)
    spearman = per_gene_pearson(true_rank, predicted_rank)

    true_variance = true.var(axis=0)
    predicted_variance = predicted.var(axis=0)
    variation_distance = np.mean(
        np.square(predicted_variance - true_variance)
        / np.maximum(np.square(true_variance), 1.0e-12)
    )

    dot = (true * predicted).sum(axis=1)
    norm = np.linalg.norm(true, axis=1) * np.linalg.norm(predicted, axis=1)
    cosine = np.divide(dot, norm, out=np.full_like(dot, np.nan), where=norm > 0)

    true_moran = per_gene_morans_i(true, coordinates)
    predicted_moran = per_gene_morans_i(predicted, coordinates)
    valid_moran = np.isfinite(true_moran) & np.isfinite(predicted_moran)
    moran_mae = (
        float(np.mean(np.abs(predicted_moran[valid_moran] - true_moran[valid_moran])))
        if valid_moran.any()
        else float("nan")
    )
    moran_correlation = (
        float(per_gene_pearson(
            true_moran[valid_moran, None], predicted_moran[valid_moran, None]
        )[0])
        if valid_moran.sum() >= 2
        else float("nan")
    )

    metrics: dict[str, float | int] = {
        "spots": int(true.shape[0]),
        "genes": int(true.shape[1]),
        "pcc_macro_all_fixed_genes": float(defined.mean()) if len(defined) else float("nan"),
        "pcc_median_all_fixed_genes": float(np.median(defined)) if len(defined) else float("nan"),
        "pcc_defined_gene_count": int(np.isfinite(pcc).sum()),
        "pcc_positive_gene_fraction": float(np.mean(defined > 0)) if len(defined) else float("nan"),
        "spearman_macro_all_fixed_genes": (
            float(np.mean(spearman[np.isfinite(spearman)]))
            if np.isfinite(spearman).any()
            else float("nan")
        ),
        "mae_log_normalized": float(absolute_error.mean()),
        "mse_log_normalized": float(np.square(predicted - true).mean()),
        "rmse_log_normalized": float(np.sqrt(np.square(predicted - true).mean())),
        "mean_spot_cosine": float(np.nanmean(cosine)),
        "relative_variation_distance": float(variation_distance),
        "prediction_negative_fraction": float(np.mean(predicted < 0)),
        "moran_i_mae": moran_mae,
        "moran_i_gene_correlation": moran_correlation,
    }

    # GenAR/Stem 按测试 PCC 排序后报告 top-k；保留该口径只是为了与论文表格对照。
    ranked = np.sort(defined)[::-1]
    for k in (10, 50, 200):
        take = min(k, len(ranked))
        metrics[f"paper_pcc_top{k}_test_selected"] = (
            float(ranked[:take].mean()) if take else float("nan")
        )
    return metrics, pcc


def evaluate_raw_counts(true: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """对直接生成原始整数计数的方法补充计数尺度指标。"""

    true, predicted = _validate(true, predicted)
    return {
        "raw_mae": float(np.abs(predicted - true).mean()),
        "raw_rmse": float(np.sqrt(np.square(predicted - true).mean())),
        "true_zero_fraction": float(np.mean(true == 0)),
        "predicted_zero_fraction": float(np.mean(predicted == 0)),
        "zero_fraction_absolute_error": float(abs(np.mean(predicted == 0) - np.mean(true == 0))),
        "non_integer_prediction_fraction": float(np.mean(predicted != np.rint(predicted))),
        "negative_count_fraction": float(np.mean(predicted < 0)),
    }
