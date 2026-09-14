#!/usr/bin/env bash
# 在 WSL 中按顺序运行剩余的 GPU 实验（共享同一块 RTX 5060，串行避免显存竞争）。
# 由上一轮会话的 ResSAT 官方重训（scripts/06）之后接管。
set -euo pipefail
cd "/mnt/d/Code/H&E Generation"
export PYTHONUNBUFFERED=1

echo "===== [pipeline] $(date '+%F %T') 06 ResSAT 官方示例重训（uint8 修复版） ====="
python3 scripts/06_train_ressat_official.py 2>&1 | tee results/logs_06_ressat_official_retrain.txt

echo "===== [pipeline] $(date '+%F %T') 08 ResSAT 官方示例评测（uint8 修复版权重） ====="
python3 scripts/08_evaluate_ressat_official.py 2>&1 | tee results/logs_08_ressat_official_eval_fixed.txt

echo "===== [pipeline] $(date '+%F %T') 12 ResSAT 统一协议训练+评测 ====="
python3 scripts/12_train_ressat_unified.py 2>&1 | tee results/logs_12_ressat_unified.txt

echo "===== [pipeline] $(date '+%F %T') 09 GenAR 统一协议训练 ====="
python3 scripts/09_train_genar.py --epochs 40 --global-batch-size 16 2>&1 | tee results/logs_09_genar_train.txt

echo "===== [pipeline] $(date '+%F %T') 10 GenAR 推理 ====="
CKPT=$(ls -t logs/gse240429/GENAR/*/best-epoch=*.ckpt 2>/dev/null | head -1)
if [ -z "$CKPT" ]; then
  echo "[pipeline] 未找到 GenAR checkpoint，跳过推理"
else
  echo "[pipeline] checkpoint: $CKPT"
  python3 scripts/10_genar_inference.py --ckpt-path "$CKPT" 2>&1 | tee results/logs_10_genar_infer.txt
fi

echo "===== [pipeline] $(date '+%F %T') 13 Stem 统一协议训练 ====="
python3 scripts/13_train_stem_adapted.py 2>&1 | tee results/logs_13_stem_train.txt

echo "===== [pipeline] $(date '+%F %T') 14 Stem 采样评测 ====="
python3 scripts/14_sample_stem_adapted.py 2>&1 | tee results/logs_14_stem_sample.txt

echo "===== [pipeline] $(date '+%F %T') 15 汇总统一对比 ====="
python3 scripts/15_make_comparison.py 2>&1 | tee results/logs_15_comparison.txt

echo "===== [pipeline] $(date '+%F %T') 全部完成 ====="
