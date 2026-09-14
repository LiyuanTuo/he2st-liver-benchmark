import sys

sys.path.insert(0, "src")
import numpy as np

from he2st.metrics import evaluate_normalized

r = np.load("results/ressat_unified/unified_batch16_predictions.npz")
true = r["true"].astype(np.float32)
predicted = r["predicted"].astype(np.float32)
coordinates = r["coordinates_xy"].astype(np.float32)
metrics, per_gene_pcc = evaluate_normalized(true, predicted, coordinates)
for key in [
    "pcc_macro_all_fixed_genes", "spearman_macro_all_fixed_genes",
    "mae_log_normalized", "rmse_log_normalized", "mean_spot_cosine",
    "relative_variation_distance", "moran_i_mae", "moran_i_gene_correlation",
]:
    print(f"{key}: {metrics[key]}")
print("per-gene pcc mean:", float(np.nanmean(per_gene_pcc)),
      "| >0 fraction:", float(np.mean(per_gene_pcc > 0)))
