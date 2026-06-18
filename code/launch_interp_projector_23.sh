#!/usr/bin/env bash
set -euo pipefail

ALPHA=${ALPHA:-0.3}
TAG=${TAG:-proj_a030}
PID=/mnt/data/cyh/interp_projector_${TAG}.pid
OUTER=/mnt/data/cyh/interp_projector_${TAG}.outer.log

if [[ -f "${PID}" ]]; then
  old_pid=$(cat "${PID}" || true)
  if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" 2>/dev/null; then
    echo "already_running pid=${old_pid}"
    exit 0
  fi
fi

nohup env ALPHA="${ALPHA}" TAG="${TAG}" bash /mnt/data/cyh/run_interp_projector_23.sh > "${OUTER}" 2>&1 &
echo $! > "${PID}"
echo "launched pid=$(cat "${PID}") alpha=${ALPHA} tag=${TAG} log=${OUTER}"
