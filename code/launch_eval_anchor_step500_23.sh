#!/usr/bin/env bash
set -euo pipefail

LOG=/mnt/data/cyh/eval_anchor_step500_23.outer.log
PID=/mnt/data/cyh/eval_anchor_step500_23.pid

if [[ -f "${PID}" ]]; then
  old_pid=$(cat "${PID}" || true)
  if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" 2>/dev/null; then
    echo "already_running pid=${old_pid}"
    exit 0
  fi
fi

nohup bash /mnt/data/cyh/eval_anchor_step500_23.sh > "${LOG}" 2>&1 &
echo $! > "${PID}"
echo "launched pid=$(cat "${PID}") log=${LOG}"
