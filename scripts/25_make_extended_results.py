"""汇总新增实验并生成三张可直接用于报告/PPT的结果图。

脚本无命令行参数。只有在 ResSAT、跨切片稳健性和 HEST-IDC 实验均完成后运行，
避免把尚未得到的数值或论文数值误写成本项目实验结果。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"


def read_json(relative: str) -> dict:
    path = ROOT / relative
    if not path.exists():
        raise FileNotFoundError(f"缺少实验结果：{path}")
    return json.loads(path.read_text(encoding="utf-8"))


def annotate_bars(axis, bars, digits=3):
    for bar in bars:
        value = bar.get_height()
        axis.text(bar.get_x() + bar.get_width() / 2, value + 0.008, f"{value:.{digits}f}",
                  ha="center", va="bottom", fontsize=9)


def plot_liver_robustness(cross: dict, unified: dict, second_fold: dict) -> None:
    methods = ["Image Ridge", "MLP ensemble", "BLEEP ensemble", "ResSAT"]
    c1 = cross["folds"]["test_C1"]["methods"]
    d1 = cross["folds"]["test_D1"]["methods"]
    c1_values = [
        c1["image_ridge"]["test_metrics"]["pcc_macro_all_fixed_genes"],
        c1["stnet_style_mlp"]["ensemble_test_metrics"]["pcc_macro_all_fixed_genes"],
        c1["bleep"]["ensemble_test_metrics"]["pcc_macro_all_fixed_genes"],
        second_fold["strict_full_expression_metrics"]["pcc_macro_all_fixed_genes"],
    ]
    d1_values = [
        d1["image_ridge"]["test_metrics"]["pcc_macro_all_fixed_genes"],
        d1["stnet_style_mlp"]["ensemble_test_metrics"]["pcc_macro_all_fixed_genes"],
        d1["bleep"]["ensemble_test_metrics"]["pcc_macro_all_fixed_genes"],
        unified["unified_table"]["ressat_unified"]["pcc_macro_all_fixed_genes"],
    ]
    x = np.arange(len(methods))
    fig, axis = plt.subplots(figsize=(10.5, 5.2))
    first = axis.bar(x - 0.18, c1_values, 0.36, label="Test slide C1", color="#3077B8")
    second = axis.bar(x + 0.18, d1_values, 0.36, label="Test slide D1", color="#E58A39")
    annotate_bars(axis, first)
    annotate_bars(axis, second)
    axis.set_xticks(x, methods)
    axis.set_ylabel("Macro PCC across 200 fixed genes")
    axis.set_title("Human liver: performance changes with the held-out slide")
    axis.set_ylim(0, 0.23)
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES / "06_cross_slide_robustness.png", dpi=220)
    plt.close(fig)


def plot_ressat_replication(sa: dict, sp: dict) -> None:
    # 最终版论文 Table 1/2 的 ResSAT 均值；复现值来自本地 evaluation.json。
    labels = ["SA: 2,000 HVGs", "SA: top-50 HEGs", "SP: 2,000 HVGs", "SP: top-50 HEGs"]
    paper = [0.6577, 0.8781, 0.6980, 0.8999]
    sa_metrics = sa["metrics_by_inference_batch_size"]["32"]
    sp_metrics = sp["paper_reconstructed_space_metrics"]
    reproduced = [
        sa_metrics["pcc_macro_all_fixed_genes"],
        sa_metrics["official_top50_high_expression_pcc"],
        sp_metrics["pcc_macro_all_fixed_genes"],
        sp_metrics["top50_high_expression_pcc"],
    ]
    x = np.arange(len(labels))
    fig, axis = plt.subplots(figsize=(11, 5.2))
    first = axis.bar(x - 0.18, paper, 0.36, label="Paper", color="#6B7280")
    second = axis.bar(x + 0.18, reproduced, 0.36, label="This project", color="#2A9D8F")
    annotate_bars(axis, first, digits=4)
    annotate_bars(axis, second, digits=4)
    axis.set_xticks(x, labels)
    axis.set_ylabel("Macro PCC")
    axis.set_title("ResSAT replication on its original mouse-brain datasets")
    axis.set_ylim(0, 1.02)
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES / "07_ressat_original_protocol.png", dpi=220)
    plt.close(fig)


def plot_hest(hest: dict) -> None:
    names = ["ResNet18\n(this project)", "ResNet50\n(this run)", "ResNet50\nHEST reference"]
    values = [
        hest["encoders"]["resnet18_shared"]["pcc_mean_across_folds"],
        hest["encoders"]["resnet50_hest"]["pcc_mean_across_folds"],
        hest["protocol"]["official_current_resnet50_reference_pcc"],
    ]
    colors = ["#3077B8", "#2A9D8F", "#6B7280"]
    fig, axis = plt.subplots(figsize=(8.5, 5.2))
    bars = axis.bar(names, values, color=colors, width=0.58)
    annotate_bars(axis, bars, digits=4)
    axis.set_ylabel("Mean macro PCC across four folds")
    axis.set_title("External HEST-IDC benchmark (50 fixed genes)")
    axis.set_ylim(0, max(values) + 0.1)
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES / "08_hest_idc_external_benchmark.png", dpi=220)
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    cross = read_json("results/cross_slide_robustness/metrics.json")
    unified = read_json("results/unified_comparison.json")
    second_fold = read_json("results/ressat_second_fold/evaluation.json")
    sa = read_json("results/ressat_official/evaluation.json")
    sp = read_json("results/ressat_sp_rebuilt/evaluation.json")
    hest = read_json("results/hest_idc/evaluation.json")
    statistical = read_json("results/statistical_comparison.json")
    plot_liver_robustness(cross, unified, second_fold)
    plot_ressat_replication(sa, sp)
    plot_hest(hest)
    strict_key = "pcc_macro_all_fixed_genes"
    c1_methods = cross["folds"]["test_C1"]["methods"]
    d1_methods = cross["folds"]["test_D1"]["methods"]
    summary = {
        "generated_from": [
            "results/unified_comparison.json",
            "results/cross_slide_robustness/metrics.json",
            "results/ressat_second_fold/evaluation.json",
            "results/ressat_official/evaluation.json",
            "results/ressat_sp_rebuilt/evaluation.json",
            "results/hest_idc/evaluation.json",
            "results/statistical_comparison.json",
        ],
        "human_liver_two_test_slides_pcc": {
            "image_ridge": [
                c1_methods["image_ridge"]["test_metrics"][strict_key],
                d1_methods["image_ridge"]["test_metrics"][strict_key],
            ],
            "mlp_ensemble": [
                c1_methods["stnet_style_mlp"]["ensemble_test_metrics"][strict_key],
                d1_methods["stnet_style_mlp"]["ensemble_test_metrics"][strict_key],
            ],
            "bleep_ensemble": [
                c1_methods["bleep"]["ensemble_test_metrics"][strict_key],
                d1_methods["bleep"]["ensemble_test_metrics"][strict_key],
            ],
            "ressat": [
                second_fold["strict_full_expression_metrics"][strict_key],
                unified["unified_table"]["ressat_unified"][strict_key],
            ],
            "order": ["test_C1", "test_D1"],
        },
        "ressat_original_protocol": {
            "paper": {"SA_2000": 0.6577, "SA_top50": 0.8781, "SP_2000": 0.6980, "SP_top50": 0.8999},
            "reproduced": {
                "SA_2000": sa["metrics_by_inference_batch_size"]["32"][strict_key],
                "SA_top50": sa["metrics_by_inference_batch_size"]["32"]["official_top50_high_expression_pcc"],
                "SP_2000": sp["paper_reconstructed_space_metrics"][strict_key],
                "SP_top50": sp["paper_reconstructed_space_metrics"]["top50_high_expression_pcc"],
                "SP_strict_full_truth": sp["strict_log_normalized_truth_metrics"][strict_key],
            },
        },
        "hest_idc": {
            encoder: {
                "pcc_mean_across_folds": values["pcc_mean_across_folds"],
                "pcc_std_across_folds": values["pcc_std_across_folds"],
            }
            for encoder, values in hest["encoders"].items()
        },
        "statistical_comparison": statistical,
    }
    (RESULTS / "extended_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("完成：results/extended_summary.json 与 figures/06、07、08")


if __name__ == "__main__":
    main()
