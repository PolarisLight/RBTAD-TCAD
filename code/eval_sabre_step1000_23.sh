#!/usr/bin/env bash
set -euo pipefail

cd /mnt/data/cyh/VLA-long-tail

export CUDA_VISIBLE_DEVICES=2,3
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export HF_HOME=/mnt/data/cyh/.cache/huggingface
export TRANSFORMERS_CACHE=/mnt/data/cyh/.cache/huggingface
export TFDS_DATA_DIR=/mnt/data/cyh/tensorflow_datasets
export PYTHONPATH=/mnt/data/cyh/VLA-long-tail/LIBERO:/mnt/data/cyh/VLA-long-tail:${PYTHONPATH:-}

RUN_ID="sabre_tail_rescue_w2_from_rbtad34075_s2000_seed7_b20"
STEP="001000"
STAMP="$(date +%Y%m%d_%H%M%S)"
SAVE_ROOT="results/sabre_tail_rescue/${RUN_ID}/1000/sabre_step1000_30trials_egl_${STAMP}"

echo "[SABRE_STEP1000_EVAL] start $(date -Is)"
echo "[SABRE_STEP1000_EVAL] save_root=${SAVE_ROOT}"

/mnt/data/cyh/envs/vla-long-tail/bin/python vla_scripts/parallel_libero_evaluator_egl.py \
  --num-trails-per-task 30 \
  --num-gpus 2 \
  --num-processes 10 \
  --task-suite-name libero_core \
  --pretrained-checkpoint "runs/sabre_tail_rescue/${RUN_ID}" \
  --unnorm_key libero_core_lt \
  --save-root "${SAVE_ROOT}" \
  --steps "${STEP}" \
  --instruction-formatting False

echo "[SABRE_STEP1000_EVAL] done $(date -Is)"
