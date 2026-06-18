#!/usr/bin/env bash
set -euo pipefail

echo "== time =="
date -Is
echo "== gpu =="
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true
echo "== train/eval pgrep =="
pgrep -af "train_rabad_full_23|eval_rabad_after_train_23|vla_scripts/train.py|parallel_libero_evaluator" || true
echo "== latest train log =="
tail -40 /mnt/data/cyh/rabad_full_23.log 2>/dev/null || true
echo "== latest eval result =="
grep -R -E "Overall success rate|Task .*success rate" /mnt/data/cyh/VLA-long-tail/results/rabad_lt_main 2>/dev/null | tail -30 || true
