#!/usr/bin/env bash
set -euo pipefail

LIMIT_STEPS=${LIMIT_STEPS:-300}
RUN_ID=${RUN_ID:-target_task9_w8_anchor_proj_s${LIMIT_STEPS}_seed7_b20}
PID=/mnt/data/cyh/${RUN_ID}.pid
OUTER=/mnt/data/cyh/${RUN_ID}.outer.log

if [[ -f "${PID}" ]]; then
  old_pid=$(cat "${PID}" || true)
  if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" 2>/dev/null; then
    echo "already_running pid=${old_pid}"
    exit 0
  fi
fi

nohup env LIMIT_STEPS="${LIMIT_STEPS}" RUN_ID="${RUN_ID}" DO_EVAL="${DO_EVAL:-0}" bash /mnt/data/cyh/run_target_task9_23.sh > "${OUTER}" 2>&1 &
echo $! > "${PID}"
echo "launched pid=$(cat "${PID}") run_id=${RUN_ID} limit_steps=${LIMIT_STEPS} log=${OUTER}"
