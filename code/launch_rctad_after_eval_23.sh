#!/usr/bin/env bash
set -euo pipefail

LOG=/mnt/data/cyh/rctad_launch_after_eval_23.log
SMOKE_LOG=/mnt/data/cyh/rctad_smoke_23.log
FULL_LOG=/mnt/data/cyh/rctad_full_23.log
SMOKE_DEBUG=/mnt/data/cyh/VLA-long-tail/runs/rctad_lt_smoke/rctad_tail9_confmedian_s5_seed7/tcad-debug.csv

{
  echo "== launcher start $(date -Is) =="
  chmod +x /mnt/data/cyh/train_rctad_smoke_23.sh /mnt/data/cyh/train_rctad_full_23.sh

  while ps -ef | grep parallel_libero_evaluator_egl | grep step15000 | grep -v grep >/dev/null; do
    echo "waiting for step15000 eval $(date -Is)"
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true
    sleep 900
  done

  echo "== step15000 eval no longer running $(date -Is) =="
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true

  echo "== launch smoke $(date -Is) =="
  bash /mnt/data/cyh/train_rctad_smoke_23.sh
  echo "== smoke exit ok $(date -Is) =="
  tail -20 "$SMOKE_LOG" || true

  if [[ ! -s "$SMOKE_DEBUG" ]]; then
    echo "smoke debug file missing: $SMOKE_DEBUG"
    exit 2
  fi

  python - <<'PY'
import csv
path = "/mnt/data/cyh/VLA-long-tail/runs/rctad_lt_smoke/rctad_tail9_confmedian_s5_seed7/tcad-debug.csv"
rows = list(csv.DictReader(open(path)))
if not rows:
    raise SystemExit("no debug rows")
candidate = sum(int(r.get("candidate_count", r.get("active_count", 0))) for r in rows)
active = sum(int(r.get("active_count", 0)) for r in rows)
print({"rows": len(rows), "candidate": candidate, "active": active})
if candidate <= 0 or active <= 0:
    raise SystemExit("RCTAD smoke produced no active counterfactual updates")
PY

  echo "== launch full $(date -Is) =="
  bash /mnt/data/cyh/train_rctad_full_23.sh
  echo "== full exit ok $(date -Is) =="
  tail -80 "$FULL_LOG" || true
} >> "$LOG" 2>&1
