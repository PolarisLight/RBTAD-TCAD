#!/usr/bin/env bash
set -euo pipefail

echo "== processes =="
ps -ef | grep parallel_libero_evaluator_egl | grep step15000 | grep -v grep || true

echo "== gpu =="
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true

echo "== results =="
grep -R -E "Overall success rate|Task .*success rate" \
  /mnt/data/cyh/VLA-long-tail/results/tcad_final_lt_main/tcad_final_maskpos_ratio025_seed7_b20/15000/tcad_final_step15000_30trials_egl_20260608_124534 \
  2>/dev/null | tail -30 || true

echo "== eval log tail =="
tail -80 /mnt/data/cyh/tcad_final_eval_step15000_egl_23.log || true
