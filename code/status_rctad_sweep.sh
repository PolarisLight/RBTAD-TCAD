#!/usr/bin/env bash
set -euo pipefail

echo "== processes =="
pgrep -af "rctad_ckpt_sweep|parallel_libero_evaluator_egl" || true

echo "== gpu =="
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true

echo "== sweep tail =="
tail -100 /mnt/data/cyh/rctad_ckpt_sweep_23.log 2>/dev/null || true

echo "== sweep summaries =="
grep -R -E "Overall success rate|Task .*success rate" \
  /mnt/data/cyh/VLA-long-tail/results/rctad_lt_main/rctad_tail9_confmedian_ratio05_seed7_b20 \
  2>/dev/null | tail -80 || true
