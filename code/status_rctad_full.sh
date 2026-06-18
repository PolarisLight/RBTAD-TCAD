#!/usr/bin/env bash
set -euo pipefail

echo "== processes =="
pgrep -af "train_rctad_full_23|vla_scripts/train.py|torchrun" || true

echo "== gpu =="
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true

echo "== log tail =="
tail -120 /mnt/data/cyh/rctad_full_23.log 2>/dev/null || true

echo "== debug tail =="
tail -20 /mnt/data/cyh/VLA-long-tail/runs/rctad_lt_main/rctad_tail9_confmedian_ratio05_seed7_b20/tcad-debug.csv 2>/dev/null || true

echo "== checkpoints =="
find /mnt/data/cyh/VLA-long-tail/runs/rctad_lt_main/rctad_tail9_confmedian_ratio05_seed7_b20/checkpoints -maxdepth 1 -type f -printf "%f %s\n" 2>/dev/null | sort || true
