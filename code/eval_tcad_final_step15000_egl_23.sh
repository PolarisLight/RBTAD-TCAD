#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/data/cyh/VLA-long-tail
ENV=/mnt/data/cyh/envs/vla-long-tail
LOG=/mnt/data/cyh/tcad_final_eval_step15000_egl_23.log
RUN_NAME=tcad_final_lt_main/tcad_final_maskpos_ratio025_seed7_b20
STEP=15000
RESULTS_NAME=tcad_final_step15000_30trials_egl_$(date +"%Y%m%d_%H%M%S")

{
  echo "== tcad-final eval step15000 egl start $(date -Is) =="
  source /opt/miniconda3/etc/profile.d/conda.sh
  conda activate "$ENV"
  cd "$ROOT"

  export CUDA_VISIBLE_DEVICES=1,3
  export MUJOCO_GL=egl
  export HF_ENDPOINT=https://hf-mirror.com
  export HF_HOME=/mnt/data/cyh/.cache/huggingface
  export PRISMATIC_DATA_ROOT="$ROOT/data/prismatic"
  export PYTHONPATH="$ROOT/LIBERO:$ROOT:${PYTHONPATH:-}"
  export TOKENIZERS_PARALLELISM=true
  mkdir -p "results/$RUN_NAME/$STEP/$RESULTS_NAME"
  : > .hf_token

  python vla_scripts/parallel_libero_evaluator_egl.py \
    --num-trails-per-task 30 \
    --num-gpus 2 \
    --num-processes 10 \
    --task-suite-name libero_core \
    --pretrained-checkpoint "runs/$RUN_NAME" \
    --unnorm_key libero_core_lt \
    --save-root "results/$RUN_NAME/$STEP/$RESULTS_NAME" \
    --steps "$STEP" \
    --instruction-formatting False

  echo "== tcad-final eval step15000 egl done $(date -Is) =="
} >> "$LOG" 2>&1
