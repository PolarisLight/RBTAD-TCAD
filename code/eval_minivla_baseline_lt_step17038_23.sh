#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/data/cyh/VLA-long-tail
ENV=/mnt/data/cyh/envs/vla-long-tail
LOG=/mnt/data/cyh/vla_eval_baseline_lt_step17038_23.log
RUN_NAME=miniVLA_libero_core_lt/prism-qwen25-dinosiglip-224px+0_5b+mx-libero-core-lt+n0+b10+x7
RESULTS_NAME=baseline_lt_step17038_30trials_$(date +"%Y%m%d_%H%M%S")

{
  echo "== baseline lt eval start $(date -Is) =="
  source /opt/miniconda3/etc/profile.d/conda.sh
  conda activate "$ENV"
  cd "$ROOT"

  export MUJOCO_GL=osmesa
  export HF_ENDPOINT=https://hf-mirror.com
  export HF_HOME=/mnt/data/cyh/.cache/huggingface
  export PRISMATIC_DATA_ROOT="$ROOT/data/prismatic"
  export PYTHONPATH="$ROOT/LIBERO:$ROOT:${PYTHONPATH:-}"
  export TOKENIZERS_PARALLELISM=true
  mkdir -p "results/$RUN_NAME/17038/$RESULTS_NAME"
  : > .hf_token

  echo "== command =="
  echo "python vla_scripts/parallel_libero_evaluator.py --num-trails-per-task 30 --num-gpus 2 --num-processes 10 --task-suite-name libero_core --pretrained-checkpoint runs/$RUN_NAME --unnorm_key libero_core_lt --save-root results/$RUN_NAME/17038/$RESULTS_NAME --steps 17038 --instruction-formatting False"

  python vla_scripts/parallel_libero_evaluator.py \
    --num-trails-per-task 30 \
    --num-gpus 2 \
    --num-processes 10 \
    --task-suite-name libero_core \
    --pretrained-checkpoint "runs/$RUN_NAME" \
    --unnorm_key libero_core_lt \
    --save-root "results/$RUN_NAME/17038/$RESULTS_NAME" \
    --steps 17038 \
    --instruction-formatting False

  echo "== baseline lt eval done $(date -Is) =="
} >> "$LOG" 2>&1
