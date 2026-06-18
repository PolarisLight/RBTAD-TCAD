#!/usr/bin/env bash
set -euo pipefail

echo "== before =="
pgrep -af "rctad_ckpt_sweep|parallel_libero_evaluator_egl" || true

pkill -f "rctad_ckpt_sweep" || true
pkill -f "parallel_libero_evaluator_egl.py --num-trails-per-task 30 --num-gpus 2 --num-processes 10 --task-suite-name libero_core --pretrained-checkpoint runs/rctad_lt_main/rctad_tail9_confmedian_ratio05_seed7_b20" || true

sleep 5

echo "== after =="
pgrep -af "rctad_ckpt_sweep|parallel_libero_evaluator_egl" || true
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true
