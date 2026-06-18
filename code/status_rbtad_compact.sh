#!/usr/bin/env bash
set -euo pipefail

RUN=/mnt/data/cyh/VLA-long-tail/runs/rbtad_lt_main/rbtad_w3_tail9_confmedian_seed7_b20

echo "== processes =="
pgrep -af "train_rbtad_full_23|vla_scripts/train.py|eval_rbtad_after_train|parallel_libero_evaluator_egl" || true

echo "== gpu =="
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true

echo "== metrics =="
for f in "$RUN"/*.jsonl "$RUN"/run-metrics.jsonl; do
  [[ -f "$f" ]] || continue
  echo "file=$f"
  tail -1 "$f" || true
done

echo "== debug summary =="
/mnt/data/cyh/envs/vla-long-tail/bin/python - <<'PY'
import csv, pathlib
path = pathlib.Path("/mnt/data/cyh/VLA-long-tail/runs/rbtad_lt_main/rbtad_w3_tail9_confmedian_seed7_b20/tcad-debug.csv")
if not path.exists():
    print("missing")
else:
    rows = list(csv.DictReader(path.open()))
    print("rows", len(rows))
    if rows:
        print("last", rows[-1])
        last = rows[-1000:]
        cand = sum(int(r.get("candidate_count", 0)) for r in last)
        active = sum(int(r.get("active_count", 0)) for r in last)
        print("last1000_candidate_mean", cand / len(last))
        print("last1000_active_mean", active / len(last))
PY

echo "== checkpoints =="
find "$RUN/checkpoints" -maxdepth 1 -type f -printf "%f %s\n" 2>/dev/null | sort || true

echo "== eval watcher =="
tail -40 /mnt/data/cyh/rbtad_eval_after_train_23.log 2>/dev/null || true
