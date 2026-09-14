#!/usr/bin/env bash
# 续跑流水线：从 GenAR 训练开始（前面步骤已完成）。
# GenAR 已修复 --allow-nondeterministic 默认值（adaptive_avg_pool2d backward 无确定性实现）。
set -euo pipefail
cd "/mnt/d/Code/H&E Generation"
export PYTHONUNBUFFERED=1

echo "===== [pipeline2] $(date '+%F %T') 09 GenAR 统一协议训练 ====="
python3 scripts/09_train_genar.py --epochs 40 --global-batch-size 16 2>&1 | tee results/logs_09_genar_train.txt

echo "===== [pipeline2] $(date '+%F %T') 10 GenAR 推理 ====="
CKPT=$(ls -t logs/gse240429/GENAR/*/best-epoch=*.ckpt 2>/dev/null | head -1)
if [ -z "$CKPT" ]; then
  echo "[pipeline2] 未找到 GenAR checkpoint，跳过推理"
else
  echo "[pipeline2] checkpoint: $CKPT"
  python3 scripts/10_genar_inference.py --ckpt-path "$CKPT" 2>&1 | tee results/logs_10_genar_infer.txt
fi

echo "===== [pipeline2] $(date '+%F %T') 13 Stem 统一协议训练 ====="
python3 scripts/13_train_stem_adapted.py 2>&1 | tee results/logs_13_stem_train.txt

echo "===== [pipeline2] $(date '+%F %T') 14 Stem 采样评测 ====="
python3 scripts/14_sample_stem_adapted.py 2>&1 | tee results/logs_14_stem_sample.txt

echo "===== [pipeline2] $(date '+%F %T') 15 汇总统一对比 ====="
python3 scripts/15_make_comparison.py 2>&1 | tee results/logs_15_comparison.txt

echo "===== [pipeline2] $(date '+%F %T') 16 预测热图与稀疏性 ====="
python3 scripts/16_make_prediction_figures.py 2>&1 | tee results/logs_16_prediction_figures.txt

echo "===== [pipeline2] $(date '+%F %T') 全部完成 ====="
