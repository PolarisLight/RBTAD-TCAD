#!/usr/bin/env bash
set -euo pipefail

pid_file=/mnt/data/cyh/tcad_lite_smoke_23.pid
log_file=/mnt/data/cyh/tcad_lite_smoke_23.log

echo "== status $(date -Is) =="
if [[ -f "$pid_file" ]]; then
  pid=$(cat "$pid_file")
  echo "PID=$pid"
  ps -fp "$pid" || true
else
  echo "PID file missing"
fi

echo "== gpu =="
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
echo "== compute apps =="
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true
echo "== log tail =="
tail -80 "$log_file" || true
