"""HEG 基准分析：量化"低表达基因拉低平均 PCC"的原因，并重算高表达基因口径。

不训练、不下载：复用 scripts/27 的加载与归一化口径，对同一批 D1 预测
按基因平均表达分层重算 PCC，输出：
- results/heg_benchmark.json        逐方法 × 基因子集的宏平均 PCC
- figures/benchmark_ppt/cause_gene_expression.png   表达水平 vs 逐基因 PCC 散点
- figures/benchmark_ppt/heg_benchmark_bars.png      子集口径对比柱状图

运行：py -3.12 scripts/32_heg_benchmark.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scipy_stats

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
ARRAYS = ROOT / "data/processed/gse240429/arrays"
ASSETS = ROOT / "figures/benchmark_ppt"
OUT_JSON = RESULTS / "heg_benchmark.json"

plt.rcParams.update({"font.family": "Microsoft YaHei", "axes.unicode_minus": False})
RED, BLUE, GOLD, INK = "#B71C1C", "#6C8EBF", "#B08D57", "#262626"

# 与 scripts/27 完全一致的方法口径：名称 -> (相对路径, npz 键, 是否原始计数)
METHODS = [
    ("Image Ridge", "unified_image_baselines/image_ridge_C73_D1.npz", "predicted", False),
    ("MLP 基线", "unified_image_baselines/stnet_style_mlp_C73_D1.npz", "predicted", False),
    ("BLEEP", "unified_image_baselines/bleep_C73_D1.npz", "predicted", False),
    ("ResSAT", "ressat_unified/unified_batch16_predictions.npz", "predicted", False),
    ("GenAR", "genar_gse240429/C73_D1_predictions.npz", "predicted_counts", True),
    ("Stem", "stem_adapted/stem_samples_D1.npz", "mean_prediction", False),
]
# 2026-09-08 改进后的预测（BLEEP 扩大邻域、GenAR top-k 校验、Stem 裁剪修复）
IMPROVED = [
    ("BLEEP(改)", "improvement_20260908/bleep/C73_D1_predictions.npz", "predicted", False),
    ("GenAR(改)", "improvement_20260908/genar/C73_D1_predictions.npz", "predicted_counts", True),
    ("Stem(改)", "improvement_20260908/stem/C73_D1_predictions.npz", "predicted", False),
]


def panel_normalize(values):
    """同 scripts/27：非负化 → 200 基因合计缩放至 10000 → log1p。"""
    values = np.maximum(np.asarray(values, dtype=np.float64), 0)
    return np.log1p(10000 * values / np.maximum(values.sum(1, keepdims=True), 1e-12))


def gene_pcc(truth, prediction):
    x, y = truth - truth.mean(0), prediction - prediction.mean(0)
    den = np.sqrt((x * x).sum(0) * (y * y).sum(0))
    valid = (np.ptp(truth, axis=0) > 1e-10) & (np.ptp(prediction, axis=0) > 1e-10)
    return np.divide((x * y).sum(0), den, out=np.full(truth.shape[1], np.nan), where=valid)


def load_slide(section):
    with np.load(ARRAYS / f"C73_{section}.npz") as data:
        raw, genes = data["raw_counts"], data["genes"]
    return raw, genes


def load_predictions():
    """返回 {名称: 面板内归一化后的 D1 预测}，真值另算。"""
    raw_d1, genes = load_slide("D1")
    truth = panel_normalize(raw_d1)
    predictions = {}
    for name, relative, key, is_count in METHODS + IMPROVED:
        with np.load(RESULTS / relative) as data:
            values = data[key].astype(np.float64)
            npz_genes = data["gene_names"] if is_count else data["genes"]
            assert np.array_equal(npz_genes, genes), name
        assert values.shape == truth.shape == (2265, 200), name
        prediction = panel_normalize(values if is_count else np.expm1(values))
        predictions[name] = prediction
    return truth, genes, predictions


def subset_ranks(expression, n=None):
    """按表达式从低到高排序的基因下标。"""
    return np.argsort(expression)[-n:] if n else np.argsort(expression)


def main():
    truth, genes, predictions = load_predictions()
    per_gene_pcc = {name: gene_pcc(truth, pred) for name, pred in predictions.items()}

    # 基因平均表达：测试真值口径（论文 top-50 HEG 做法）与训练集口径（不泄漏测试）
    mean_test = truth.mean(axis=0)
    mean_train = np.concatenate(
        [panel_normalize(load_slide(s)[0]) for s in ("A1", "B1")], axis=0
    ).mean(axis=0)

    order_test = np.argsort(mean_test)   # 升序
    order_train = np.argsort(mean_train)

    def macro_pcc(indices):
        return {name: float(np.nanmean(pcc[indices])) for name, pcc in per_gene_pcc.items()}

    subsets = {
        "all_200_genes": np.arange(200),
        "top50_heg_test_selected": order_test[-50:],
        "top50_heg_train_selected": order_train[-50:],
        "top100_heg_test_selected": order_test[-100:],
        "top20_heg_test_selected": order_test[-20:],
        "top10_heg_test_selected": order_test[-10:],
        "bottom50_low_expression": order_test[:50],
    }
    table = {key: macro_pcc(idx) for key, idx in subsets.items()}

    # 表达水平与逐基因 PCC 的关系（全部 200 个基因，Spearman）
    relation = {}
    for name, pcc in per_gene_pcc.items():
        mask = np.isfinite(pcc)
        rho, p_value = scipy_stats.spearmanr(mean_test[mask], pcc[mask])
        relation[name] = {"spearman_rho": float(rho), "p_value": float(p_value)}

    # 高低表达两半的对比
    low_half, high_half = order_test[:100], order_test[100:]
    split = {
        name: {
            "low_half_mean_pcc": float(np.nanmean(pcc[low_half])),
            "high_half_mean_pcc": float(np.nanmean(pcc[high_half])),
        }
        for name, pcc in per_gene_pcc.items()
    }

    output = {
        "protocol": {
            "dataset": "GSE240429 C73_D1",
            "evaluation_space": "log1p(10000 * 非负丰度 / 200 基因面板行合计)，同 scripts/27",
            "heg_selection": {
                "test_selected": "按 D1 测试真值平均表达排序（ResSAT 论文 top-50 HEG 口径）",
                "train_selected": "按 A1+B1 训练真值平均表达排序（不泄漏测试标签）",
            },
            "note": "全部复用已有预测文件，未重新训练。HEG 子集从固定 200 基因面板内选取。",
        },
        "per_gene_mean_expression_test": dict(zip(genes.tolist(), mean_test.tolist())),
        "per_gene_mean_expression_train": dict(zip(genes.tolist(), mean_train.tolist())),
        "subset_pcc_table": table,
        "expression_pcc_spearman": relation,
        "low_vs_high_half": split,
        "top50_heg_test_genes": genes[order_test[-50:]].tolist(),
        "top50_heg_train_genes": genes[order_train[-50:]].tolist(),
    }
    (ROOT / OUT_JSON).write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(relation, ensure_ascii=False, indent=2))
    print(json.dumps(split, ensure_ascii=False, indent=2))
    print(json.dumps(table, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
