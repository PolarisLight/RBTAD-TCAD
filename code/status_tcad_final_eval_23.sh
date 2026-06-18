#!/usr/bin/env bash
set -euo pipefail

log_file=/mnt/data/cyh/tcad_final_eval_step34075_egl_23.log
root=/mnt/data/cyh/VLA-long-tail/results/tcad_final_lt_main/tcad_final_maskpos_ratio025_seed7_b20/34075

echo "== status $(date -Is) =="
echo "== procs =="
ps -ef | grep parallel_libero_evaluator_egl | grep tcad_final | grep -v grep || true
echo "== gpu =="
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
echo "== result dirs =="
find "$root" -maxdepth 3 -type f -name "000.log" -printf "%p %s\n" 2>/dev/null | sort || true
echo "== recent success lines =="
grep -R "success\\|Success\\|overall\\|Overall" -n "$root" 2>/dev/null | tail -30 || true
echo "== log tail =="
tail -120 "$log_file" || true
