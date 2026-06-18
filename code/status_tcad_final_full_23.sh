#!/usr/bin/env bash
set -euo pipefail

pid_file=/mnt/data/cyh/tcad_final_full_23.pid
log_file=/mnt/data/cyh/tcad_final_full_23.log
run_dir=/mnt/data/cyh/VLA-long-tail/runs/tcad_final_lt_main/tcad_final_maskpos_ratio025_seed7_b20

echo "== status $(date -Is) =="
if [[ -f "$pid_file" ]]; then
  pid=$(cat "$pid_file")
  echo "PID=$pid"
  ps -fp "$pid" || true
else
  echo "PID file missing"
fi
echo "== torchrun =="
ps -ef | grep torchrun | grep tcad_final_maskpos_ratio025_seed7_b20 | grep -v grep || true
echo "== gpu =="
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
echo "== files =="
find "$run_dir" -maxdepth 3 -type f -printf "%p %s\n" 2>/dev/null | sort || true
echo "== tcad debug tail =="
tail -10 "$run_dir/tcad-debug.csv" 2>/dev/null || true
echo "== metrics tail =="
tail -5 "$run_dir/tcad_final_maskpos_ratio025_seed7_b20.jsonl" 2>/dev/null || true
echo "== log tail =="
tail -80 "$log_file" || true
