"""分析已完成基线的结果（非图像 vs 图像），供最终报告讨论部分使用。"""

import json

import numpy as np

nonimage = json.load(
    open("results/nonimage_baselines/metrics.json", encoding="utf-8")
)["test_metrics"]
image = json.load(
    open("results/unified_image_baselines/metrics.json", encoding="utf-8")
)["test_metrics"]

print("方法 | PCC macro | Spearman | Moran corr | RVD | spot cos")
for name, metrics in {**nonimage, **image}.items():
    def fmt(value):
        return f"{value:+.4f}" if isinstance(value, float) else "   None"

    print(
        f"{name:20s} {fmt(metrics['pcc_macro_all_fixed_genes'])} "
        f"{fmt(metrics['spearman_macro_all_fixed_genes'])} "
        f"{fmt(metrics['moran_i_gene_correlation'])} "
        f"{metrics['relative_variation_distance']:.4f} "
        f"{metrics['mean_spot_cosine']:.4f}"
    )

best_nonimage = max(
    nonimage[n]["pcc_macro_all_fixed_genes"]
    for n in nonimage
    if nonimage[n]["pcc_macro_all_fixed_genes"] is not None
)
best_image = max(image[n]["pcc_macro_all_fixed_genes"] for n in image)
print(f"\n图像最优 - 非图像最优 = {best_image - best_nonimage:+.4f}")

data = np.load("results/unified_image_baselines/image_ridge_C73_D1.npz")
pcc = data["per_gene_pcc"]
print(
    "image_ridge 每基因 PCC: mean",
    round(float(np.nanmean(pcc)), 4),
    "median",
    round(float(np.nanmedian(pcc)), 4),
    "| >0.2 比例",
    round(float(np.mean(pcc > 0.2)), 3),
    "| >0 比例",
    round(float(np.mean(pcc > 0)), 3),
)
# 基线中可预测性最高的 10 个基因
genes = [str(gene) for gene in data["genes"]]
order = np.argsort(-pcc)
print("PCC 最高 10 基因:", ", ".join(f"{genes[i]}({pcc[i]:.2f})" for i in order[:10]))
print("PCC 最低 5 基因:", ", ".join(f"{genes[i]}({pcc[i]:.2f})" for i in order[-5:]))
