"""在两张从未参与基因选择的人肝切片上复测图像方法。

固定训练 A1+B1；C1/D1 轮流充当验证集和测试集。MLP 与 BLEEP 各运行
5 个随机种子，并额外评测 5 模型平均预测。这样既避免测试切片参与早停，
也能区分一次随机初始化的偶然结果与稳定趋势。
"""

from __future__ import annotations

import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location("image_baselines", ROOT / "scripts/07_train_image_baselines.py")
baseline = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(baseline)

from he2st.metrics import evaluate_normalized


OUT = ROOT / "results/cross_slide_robustness"
SEEDS = (11, 22, 33, 44, 55)
FOLDS = {"test_C1": ("C73_D1", "C73_C1"), "test_D1": ("C73_C1", "C73_D1")}


def clean(value):
    """递归转换 NumPy 数字及 NaN，保证输出是标准 JSON。"""
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    return None if isinstance(value, float) and not np.isfinite(value) else value


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def summarize(runs: list[dict]) -> dict:
    """对每个数值指标给出五次运行均值和样本标准差。"""
    keys = runs[0].keys()
    return {
        key: {
            "mean": float(np.mean([run[key] for run in runs])),
            "std": float(np.std([run[key] for run in runs], ddof=1)),
        }
        for key in keys
        if isinstance(runs[0][key], (int, float)) and key not in {"spots", "genes", "pcc_defined_gene_count"}
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_parts = [baseline.load_slide(name) for name in ("C73_A1", "C73_B1")]
    train_x = np.concatenate([part["features"] for part in train_parts]).astype(np.float32)
    train_y = np.concatenate([part["log_normalized"] for part in train_parts]).astype(np.float32)

    # 标准化参数只能由训练切片估计。
    mean = train_x.mean(0, keepdims=True)
    std = np.maximum(train_x.std(0, keepdims=True), 1e-6)
    train_x = (train_x - mean) / std
    train_set = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))

    output = {
        "protocol": {
            "train": ["C73_A1", "C73_B1"],
            "folds": FOLDS,
            "fixed_genes": 200,
            "seeds": list(SEEDS),
            "selection_rule": "validation slide selects early stopping / Ridge alpha / BLEEP K; test labels used once",
        },
        "folds": {},
    }

    for fold_name, (validation_name, test_name) in FOLDS.items():
        print(f"\n=== {fold_name}: validation={validation_name}, test={test_name} ===")
        validation, test = baseline.load_slide(validation_name), baseline.load_slide(test_name)
        validation_x = (validation["features"].astype(np.float32) - mean) / std
        test_x = (test["features"].astype(np.float32) - mean) / std
        validation_y = validation["log_normalized"].astype(np.float32)
        validation_tensors = (torch.from_numpy(validation_x), torch.from_numpy(validation_y))
        methods: dict[str, object] = {}

        # Ridge 没有随机性；alpha 仍只由该折验证片选择。
        candidates = {}
        for alpha in (1.0, 10.0, 100.0, 1000.0):
            candidate = Ridge(alpha=alpha).fit(train_x, train_y)
            candidates[str(alpha)] = evaluate_normalized(
                validation_y, candidate.predict(validation_x), validation["coordinates_xy"]
            )[0]["pcc_macro_all_fixed_genes"]
        selected_alpha = float(max(candidates, key=candidates.get))
        ridge_prediction = Ridge(alpha=selected_alpha).fit(train_x, train_y).predict(test_x).astype(np.float32)
        ridge_metrics = evaluate_normalized(test["log_normalized"], ridge_prediction, test["coordinates_xy"])[0]
        methods["image_ridge"] = {"selected_alpha": selected_alpha, "validation_pcc": candidates, "test_metrics": ridge_metrics}

        run_metrics = {"stnet_style_mlp": [], "bleep": []}
        run_predictions = {"stnet_style_mlp": [], "bleep": []}
        selections = {"stnet_style_mlp": [], "bleep": []}
        for seed in SEEDS:
            print(f"\nseed={seed} / MLP")
            seed_everything(seed)
            loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=0)
            mlp = baseline.ExpressionMLP(train_y.shape[1]).to(device)
            mlp, info = baseline.train_with_early_stopping(
                mlp, loader, validation_tensors, lambda model, x, y: F.mse_loss(model(x), y)
            )
            prediction = baseline.predict_mlp(mlp, test_x, device)
            run_predictions["stnet_style_mlp"].append(prediction)
            run_metrics["stnet_style_mlp"].append(
                evaluate_normalized(test["log_normalized"], prediction, test["coordinates_xy"])[0]
            )
            selections["stnet_style_mlp"].append({"seed": seed, "epochs": info["epochs_run"], "best_validation_loss": info["best_validation_loss"]})

            print(f"seed={seed} / BLEEP")
            seed_everything(seed)
            loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=0)
            bleep = baseline.FrozenBLEEP(train_y.shape[1]).to(device)
            bleep, info = baseline.train_with_early_stopping(bleep, loader, validation_tensors, lambda model, x, y: model(x, y))
            _, reference = baseline.bleeper_embeddings(bleep, train_x, train_y, device)
            with torch.inference_mode():
                validation_query = F.normalize(bleep.image_projection(torch.from_numpy(validation_x).to(device)), dim=-1).cpu().numpy()
                test_query = F.normalize(bleep.image_projection(torch.from_numpy(test_x).to(device)), dim=-1).cpu().numpy()
            neighbor_pcc = {}
            for neighbors in (1, 5, 10, 20, 50):
                val_prediction = baseline.retrieve_expression(validation_query, reference, train_y, neighbors)
                neighbor_pcc[str(neighbors)] = evaluate_normalized(
                    validation_y, val_prediction, validation["coordinates_xy"]
                )[0]["pcc_macro_all_fixed_genes"]
            selected_neighbors = int(max(neighbor_pcc, key=neighbor_pcc.get))
            prediction = baseline.retrieve_expression(test_query, reference, train_y, selected_neighbors)
            run_predictions["bleep"].append(prediction)
            run_metrics["bleep"].append(
                evaluate_normalized(test["log_normalized"], prediction, test["coordinates_xy"])[0]
            )
            selections["bleep"].append({"seed": seed, "epochs": info["epochs_run"], "best_validation_loss": info["best_validation_loss"], "selected_neighbors": selected_neighbors, "validation_pcc": neighbor_pcc})

        for method in ("stnet_style_mlp", "bleep"):
            ensemble = np.mean(run_predictions[method], axis=0).astype(np.float32)
            methods[method] = {
                "per_seed": [{"seed": seed, "metrics": metrics} for seed, metrics in zip(SEEDS, run_metrics[method])],
                "mean_std": summarize(run_metrics[method]),
                "selection": selections[method],
                "ensemble_test_metrics": evaluate_normalized(test["log_normalized"], ensemble, test["coordinates_xy"])[0],
            }
            np.savez_compressed(OUT / f"{method}_{test_name}_ensemble.npz", predicted=ensemble, true=test["log_normalized"], coordinates_xy=test["coordinates_xy"], genes=test["genes"])
        np.savez_compressed(OUT / f"image_ridge_{test_name}.npz", predicted=ridge_prediction, true=test["log_normalized"], coordinates_xy=test["coordinates_xy"], genes=test["genes"])
        output["folds"][fold_name] = {"validation": validation_name, "test": test_name, "methods": methods}

    (OUT / "metrics.json").write_text(json.dumps(clean(output), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n完成：{OUT / 'metrics.json'}")


if __name__ == "__main__":
    main()
