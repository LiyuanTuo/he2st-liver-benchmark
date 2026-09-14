#!/usr/bin/env bash
# 第三段流水线：Stem（AMP 修复后）→ 采样 → 汇总 → 预测图。
set -euo pipefail
cd "/mnt/d/Code/H&E Generation"
export PYTHONUNBUFFERED=1

echo "===== [pipeline3] $(date '+%F %T') 13 Stem 统一协议训练（AMP/batch32/300ep） ====="
python3 scripts/13_train_stem_adapted.py 2>&1 | tee results/logs_13_stem_train.txt

echo "===== [pipeline3] $(date '+%F %T') 14 Stem 采样评测 ====="
python3 scripts/14_sample_stem_adapted.py 2>&1 | tee results/logs_14_stem_sample.txt

echo "===== [pipeline3] $(date '+%F %T') 15 汇总统一对比 ====="
python3 scripts/15_make_comparison.py 2>&1 | tee results/logs_15_comparison.txt

echo "===== [pipeline3] $(date '+%F %T') 16 预测热图与稀疏性 ====="
python3 scripts/16_make_prediction_figures.py 2>&1 | tee results/logs_16_prediction_figures.txt

echo "===== [pipeline3] $(date '+%F %T') 全部完成 ====="
