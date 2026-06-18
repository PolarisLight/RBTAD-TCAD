#!/usr/bin/env bash
set -euo pipefail

echo "== launcher =="
pgrep -af "launch_rctad_after_eval_23|train_rctad|vla_scripts/train.py|parallel_libero_evaluator_egl" || true

echo "== gpu =="
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true

echo "== step15000 summary =="
grep -R -E "Overall success rate|Task .*success rate" \
  /mnt/data/cyh/VLA-long-tail/results/tcad_final_lt_main/tcad_final_maskpos_ratio025_seed7_b20/15000/tcad_final_step15000_30trials_egl_20260608_124534 \
  2>/dev/null | tail -30 || true

echo "== launcher tail =="
tail -80 /mnt/data/cyh/rctad_launch_after_eval_23.log 2>/dev/null || true

echo "== smoke tail =="
tail -80 /mnt/data/cyh/rctad_smoke_23.log 2>/dev/null || true

echo "== full tail =="
tail -80 /mnt/data/cyh/rctad_full_23.log 2>/dev/null || true
