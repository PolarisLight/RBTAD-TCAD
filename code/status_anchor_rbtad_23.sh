#!/usr/bin/env bash
set -euo pipefail

RUN_ID="anchor_rbtad_l2all_w125_a005_tail9_s500_seed7_b20"
ROOT="/mnt/data/cyh/VLA-long-tail"
LOG="/mnt/data/cyh/${RUN_ID}.log"
OUTER="/mnt/data/cyh/${RUN_ID}.outer.log"

echo "== status $(date -Is) =="
/mnt/data/cyh/envs/vla-long-tail/bin/python - <<'PY'
import subprocess

needles = [
    "anchor_rbtad_l2all_w125_a005_tail9_s500_seed7_b20",
    "parallel_libero_evaluator_egl.py",
]
out = subprocess.check_output(["ps", "-u", "cyh", "-o", "pid=", "-o", "etime=", "-o", "cmd="], text=True)
for line in out.splitlines():
    if any(item in line for item in needles):
        print(line)
PY

echo "== gpu =="
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits

echo "== train log tail =="
tail -100 "${LOG}" 2>/dev/null || true

echo "== outer log tail =="
tail -40 "${OUTER}" 2>/dev/null || true

echo "== checkpoints =="
find "${ROOT}/runs/anchor_rbtad/${RUN_ID}/checkpoints" -maxdepth 1 -type f -name 'step-*.pt' -printf "%f %s\n" 2>/dev/null | sort || true

echo "== eval logs =="
find "${ROOT}/results/anchor_rbtad/${RUN_ID}" -maxdepth 6 -type f -name "000.log" -print 2>/dev/null | sort | tail -5 || true
