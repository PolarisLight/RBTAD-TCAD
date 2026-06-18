#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/data/cyh/VLA-long-tail
ENV=/mnt/data/cyh/envs/vla-long-tail
RUN_ID=rbtad_w3_tail9_confmedian_seed7_b20
RUN_ROOT=runs/rbtad_lt_main/${RUN_ID}
STEP=30000
LOG=/mnt/data/cyh/rbtad_eval_step${STEP}_23.log

{
  echo "===== RBTAD checkpoint eval step ${STEP} ====="
  date
  source /opt/miniconda3/etc/profile.d/conda.sh
  conda activate "${ENV}"
  cd "${ROOT}"

  export CUDA_VISIBLE_DEVICES=2,3
  export HF_ENDPOINT=https://hf-mirror.com
  export HF_HOME=/mnt/data/cyh/.cache/huggingface
  export TOKENIZERS_PARALLELISM=false
  export WANDB_DISABLED=true
  export PRISMATIC_DATA_ROOT="${ROOT}/data/prismatic"
  export PYTHONPATH="${ROOT}/LIBERO:${ROOT}:${PYTHONPATH:-}"

  : > .hf_token
  SAVE_ROOT="results/rbtad_lt_main/${RUN_ID}/${STEP}/rbtad_step${STEP}_30trials_egl_$(date +%Y%m%d_%H%M%S)"
  echo "save_root=${SAVE_ROOT}"
  python vla_scripts/parallel_libero_evaluator_egl.py \
    --num-trails-per-task 30 \
    --num-gpus 2 \
    --num-processes 10 \
    --task-suite-name libero_core \
    --pretrained-checkpoint "${RUN_ROOT}" \
    --unnorm_key libero_core_lt \
    --save-root "${SAVE_ROOT}" \
    --steps "${STEP}" \
    --instruction-formatting False

  echo "===== parsed result ====="
  grep -R -E "Overall success rate|Task .*success rate" "${SAVE_ROOT}" 2>/dev/null || true
  date
} >> "${LOG}" 2>&1
