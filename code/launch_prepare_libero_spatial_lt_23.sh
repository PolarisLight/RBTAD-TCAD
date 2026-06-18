#!/usr/bin/env bash
set -euo pipefail

RUN_ID=prepare_libero_spatial_lt_23
PID=/mnt/data/cyh/${RUN_ID}.pid
OUTER=/mnt/data/cyh/${RUN_ID}.outer.log

if [[ -f "${PID}" ]]; then
  old_pid=$(cat "${PID}" || true)
  if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" 2>/dev/null; then
    echo "already_running pid=${old_pid}"
    exit 0
  fi
fi

nohup bash /mnt/data/cyh/run_prepare_libero_spatial_lt_23.sh > "${OUTER}" 2>&1 &
echo $! > "${PID}"
echo "launched pid=$(cat "${PID}") run_id=${RUN_ID} log=/mnt/data/cyh/prepare_libero_spatial_lt_23.log"
