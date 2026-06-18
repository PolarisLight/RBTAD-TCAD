#!/usr/bin/env bash
set -euo pipefail

echo "SCRIPT_LOG_TAIL"
tail -80 /mnt/data/cyh/eval_sabre_step1000_23.log || true

echo "PROC_CHECK"
ps -u cyh -o pid,etime,cmd | grep -E "eval_sabre_step1000|parallel_libero_evaluator_egl.py" | grep -v grep || true

echo "GPU_CHECK"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader

echo "EVAL_DIRS"
find /mnt/data/cyh/VLA-long-tail/results/sabre_tail_rescue/sabre_tail_rescue_w2_from_rbtad34075_s2000_seed7_b20/1000 -maxdepth 4 -type f -name "000.log" -print 2>/dev/null | sort | tail -5 || true
