#!/usr/bin/env bash
set -euo pipefail

EVAL_LOG="/mnt/data/cyh/VLA-long-tail/results/sabre_tail_rescue/sabre_tail_rescue_w2_from_rbtad34075_s2000_seed7_b20/002000/sabre_step002000_30trials_egl_20260611_105118/libero_core-prismatic/step_2000-vqa_False/000.log"

echo "PARSE_BEGIN"
/mnt/data/cyh/envs/vla-long-tail/bin/python /mnt/data/cyh/parse_eval_log.py "$EVAL_LOG"
echo "PARSE_END"

echo "TAIL_BEGIN"
tail -80 "$EVAL_LOG"
echo "TAIL_END"

echo "PROC_CHECK"
ps -u cyh -o pid,etime,cmd | grep -E "sabre|parallel_libero|train.py" | grep -v grep || true

echo "GPU_CHECK"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
