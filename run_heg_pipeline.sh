#!/usr/bin/env bash
# HEG 面板重训流水线：顺序执行，逐步写日志。
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
mkdir -p results
LOG="results/logs_heg_pipeline_$(date +%Y%m%d_%H%M%S).txt"

run() {
    echo "=== STEP $1: $2 ===" | tee -a "$LOG"
    python3 -u "$3" 2>&1 | tee -a "$LOG"
    echo "STEP_$1_OK" | tee -a "$LOG"
}

run 35 "image baselines (Ridge/MLP/BLEEP)" "scripts/35_train_heg_image_baselines.py"
run 37 "ResSAT unified train+eval" "scripts/37_train_heg_ressat.py"
run 38 "GenAR train" "scripts/38_train_heg_genar.py"
run 40 "GenAR inference D1" "scripts/40_infer_heg_genar.py"
run 39 "Stem train" "scripts/39_train_heg_stem.py"
run 41 "BLEEP validation selection" "scripts/41_bleep_heg_improved.py"
run 43 "GenAR validation-selected expected decoding" "scripts/43_genar_heg_decode.py"
run 42 "Stem validation-selected sampling" "scripts/42_stem_heg_sampling.py"
HE2ST_PANEL=gse240429_heg run 29 "ST-Net DenseNet121 HEG training" "scripts/29_train_stnet_densenet.py"
run 50 "Asset and gene audit" "scripts/50_audit_protocol_and_assets.py"
run 50b "Image alignment audit" "scripts/50b_audit_image_alignment.py"
run 51 "Fresh ResSAT checkpoint verification" "scripts/51_verify_ressat_checkpoints.py"
run 52 "Final fixed-HEG evaluation" "scripts/52_evaluate_verified_benchmark.py"

echo "=== PIPELINE DONE ===" | tee -a "$LOG"
