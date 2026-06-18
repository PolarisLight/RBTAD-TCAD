#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/data/cyh/VLA-long-tail
ENV=/mnt/data/cyh/envs/vla-long-tail
RUN_ID=anchor_rbtad_l2all_w125_a005_tail9_s500_seed7_b20
RUN_ROOT=runs/anchor_rbtad/${RUN_ID}
LOG=/mnt/data/cyh/eval_anchor_step500_23.log

{
  echo "== Anchor-RBTAD step500 eval restart $(date -Is) =="
  source /opt/miniconda3/etc/profile.d/conda.sh
  conda activate "$ENV"
  cd "$ROOT"

  unset CUDA_VISIBLE_DEVICES
  export MUJOCO_GL=egl
  export PYTHONUNBUFFERED=1
  export HF_ENDPOINT=https://hf-mirror.com
  export HF_HOME=/mnt/data/cyh/.cache/huggingface
  export TRANSFORMERS_CACHE=/mnt/data/cyh/.cache/huggingface
  export HF_HUB_DISABLE_TELEMETRY=1
  export TFDS_DATA_DIR=/mnt/data/cyh/tensorflow_datasets
  export PYTHONPATH="$ROOT/LIBERO:$ROOT:${PYTHONPATH:-}"

  mkdir -p "${RUN_ROOT}/checkpoints"
  if [[ -f "${RUN_ROOT}/checkpoints/step-000500-epoch-00-loss=0.0932.pt" && ! -e "${RUN_ROOT}/checkpoints/step-500-epoch-00-loss=0.0932.pt" ]]; then
    ln -s step-000500-epoch-00-loss=0.0932.pt "${RUN_ROOT}/checkpoints/step-500-epoch-00-loss=0.0932.pt"
  fi
  if [[ -f "${RUN_ROOT}/checkpoints/step-000500-epoch-00-loss=0.0932.pt" && ! -e "${RUN_ROOT}/checkpoints/step-00500-epoch-00-loss=0.0932.pt" ]]; then
    ln -s step-000500-epoch-00-loss=0.0932.pt "${RUN_ROOT}/checkpoints/step-00500-epoch-00-loss=0.0932.pt"
  fi

  stamp=$(date +%Y%m%d_%H%M%S)
  save_root="results/anchor_rbtad/${RUN_ID}/500/anchor_step500_30trials_egl_${stamp}"

  echo "save_root=${save_root}"
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true
  python vla_scripts/parallel_libero_evaluator_egl.py \
    --num-trails-per-task 30 \
    --num-gpus 2 \
    --num-processes 10 \
    --task-suite-name libero_core \
    --pretrained-checkpoint "${RUN_ROOT}" \
    --unnorm_key libero_core_lt \
    --save-root "${save_root}" \
    --steps 500 \
    --instruction-formatting False

  echo "== Anchor-RBTAD step500 eval done $(date -Is) =="
  grep -R -E "Overall success rate|Task .*success rate" "${save_root}" 2>/dev/null || true
} >> "${LOG}" 2>&1
