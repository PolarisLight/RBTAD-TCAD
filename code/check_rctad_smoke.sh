#!/usr/bin/env bash
set -euo pipefail

RUN=/mnt/data/cyh/VLA-long-tail/runs/rctad_lt_smoke/rctad_tail9_confmedian_s5_seed7

echo "== debug =="
cat "$RUN/tcad-debug.csv" 2>/dev/null || true

echo "== files =="
find "$RUN" -maxdepth 3 -type f -printf "%p %s\n" 2>/dev/null | sort || true

echo "== procs =="
pgrep -af "launch_rctad_after_eval_23|train_rctad|vla_scripts/train.py|parallel_libero_evaluator_egl" || true

echo "== gpu =="
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true
