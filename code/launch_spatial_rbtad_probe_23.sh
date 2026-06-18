#!/usr/bin/env bash
set -euo pipefail

LIMIT_STEPS=${LIMIT_STEPS:-1000}
NUM_TRIALS=${NUM_TRIALS:-10}
SUITE=${SUITE:-libero_spatial}
DATA_MIX=${DATA_MIX:-libero_spatial}
UNNORM_KEY=${UNNORM_KEY:-libero_spatial}
PROBE_NAME=${PROBE_NAME:-${DATA_MIX}_rbtad_probe}
RUN_ID=${PROBE_NAME}_s${LIMIT_STEPS}_t${NUM_TRIALS}
PID=/mnt/data/cyh/${RUN_ID}.pid
OUTER=/mnt/data/cyh/${RUN_ID}.outer.log

if [[ -f "${PID}" ]]; then
  old_pid=$(cat "${PID}" || true)
  if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" 2>/dev/null; then
    echo "already_running pid=${old_pid}"
    exit 0
  fi
fi

nohup env LIMIT_STEPS="${LIMIT_STEPS}" NUM_TRIALS="${NUM_TRIALS}" SUITE="${SUITE}" DATA_MIX="${DATA_MIX}" UNNORM_KEY="${UNNORM_KEY}" PROBE_NAME="${PROBE_NAME}" bash /mnt/data/cyh/run_spatial_rbtad_probe_23.sh > "${OUTER}" 2>&1 &
echo $! > "${PID}"
echo "launched pid=$(cat "${PID}") run_id=${RUN_ID} suite=${SUITE} data_mix=${DATA_MIX} unnorm=${UNNORM_KEY} steps=${LIMIT_STEPS} trials=${NUM_TRIALS} log=/mnt/data/cyh/${RUN_ID}.log"
