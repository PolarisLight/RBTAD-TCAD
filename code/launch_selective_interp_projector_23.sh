#!/usr/bin/env bash
set -euo pipefail

ALPHA=${ALPHA:-0.2}
TAG=${TAG:-p24_a020}
INCLUDE_PREFIXES=${INCLUDE_PREFIXES:-model.projector.projector.2,model.projector.projector.4}
RUN_ID=selective_interp_projector_${TAG}_seed7_b20
PID=/mnt/data/cyh/${RUN_ID}.pid
OUTER=/mnt/data/cyh/${RUN_ID}.outer.log

if [[ -f "${PID}" ]]; then
  old_pid=$(cat "${PID}" || true)
  if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" 2>/dev/null; then
    echo "already_running pid=${old_pid}"
    exit 0
  fi
fi

nohup env ALPHA="${ALPHA}" TAG="${TAG}" INCLUDE_PREFIXES="${INCLUDE_PREFIXES}" bash /mnt/data/cyh/run_selective_interp_projector_23.sh > "${OUTER}" 2>&1 &
echo $! > "${PID}"
echo "launched pid=$(cat "${PID}") run_id=${RUN_ID} alpha=${ALPHA} include=${INCLUDE_PREFIXES} log=${OUTER}"
