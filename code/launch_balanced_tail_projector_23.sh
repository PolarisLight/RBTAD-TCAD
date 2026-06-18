#!/usr/bin/env bash
set -euo pipefail

LIMIT_STEPS=${LIMIT_STEPS:-100}
TARGET_WEIGHT=${TARGET_WEIGHT:-8.0}
RUN_ID=${RUN_ID:-balanced_tail_projector_t7t9_w${TARGET_WEIGHT}_s${LIMIT_STEPS}_seed7_b20}
PID=/mnt/data/cyh/${RUN_ID}.pid
OUTER=/mnt/data/cyh/${RUN_ID}.outer.log

if [[ -f "${PID}" ]]; then
  old_pid=$(cat "${PID}" || true)
  if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" 2>/dev/null; then
    echo "already_running pid=${old_pid}"
    exit 0
  fi
fi

nohup env LIMIT_STEPS="${LIMIT_STEPS}" TARGET_WEIGHT="${TARGET_WEIGHT}" RUN_ID="${RUN_ID}" DO_EVAL="${DO_EVAL:-0}" bash /mnt/data/cyh/run_balanced_tail_projector_23.sh > "${OUTER}" 2>&1 &
echo $! > "${PID}"
echo "launched pid=$(cat "${PID}") run_id=${RUN_ID} limit_steps=${LIMIT_STEPS} target_weight=${TARGET_WEIGHT} log=${OUTER}"
