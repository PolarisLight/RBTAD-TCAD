#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/data/cyh/VLA-long-tail
ENV=/mnt/data/cyh/envs/vla-long-tail
ALPHA=${ALPHA:-0.3}
TAG=${TAG:-proj_a030}
RUN_ID=interp_projector_${TAG}_seed7_b20
RUN_ROOT=${ROOT}/runs/interp_projector/${RUN_ID}
OUT=${RUN_ROOT}/checkpoints/step-000000-epoch-00-loss=0.0000.pt
BASE=${ROOT}/runs/rbtad_lt_main/rbtad_w3_tail9_confmedian_seed7_b20/checkpoints/step-034075-epoch-36-loss=0.0678.pt
DELTA=${ROOT}/runs/anchor_rbtad/anchor_rbtad_l2all_w125_a005_tail9_s500_seed7_b20/checkpoints/step-000500-epoch-00-loss=0.0932.pt
CONFIG=${ROOT}/runs/rbtad_lt_main/rbtad_w3_tail9_confmedian_seed7_b20/config.json
STATS=${ROOT}/runs/rbtad_lt_main/rbtad_w3_tail9_confmedian_seed7_b20/dataset_statistics.json
LOG=/mnt/data/cyh/${RUN_ID}.log

{
  echo "== Projector-only interp start $(date -Is) alpha=${ALPHA} =="
  source /opt/miniconda3/etc/profile.d/conda.sh
  conda activate "$ENV"
  cd "$ROOT"

  if [[ ! -f "${OUT}" ]]; then
    /mnt/data/cyh/envs/vla-long-tail/bin/python /mnt/data/cyh/make_interp_ckpt.py \
      --base "${BASE}" \
      --delta "${DELTA}" \
      --out "${OUT}" \
      --alpha "${ALPHA}" \
      --copy-config-from "${CONFIG}" \
      --copy-stats-from "${STATS}" \
      --run-root "${RUN_ROOT}" \
      --include-prefix model.projector
  else
    echo "checkpoint exists: ${OUT}"
    cp -f "${CONFIG}" "${RUN_ROOT}/config.json"
    cp -f "${STATS}" "${RUN_ROOT}/dataset_statistics.json"
  fi

  ln -sf step-000000-epoch-00-loss=0.0000.pt "${RUN_ROOT}/checkpoints/step-00000-epoch-00-loss=0.0000.pt"

  unset CUDA_VISIBLE_DEVICES
  export MUJOCO_GL=egl
  export PYTHONUNBUFFERED=1
  export HF_ENDPOINT=https://hf-mirror.com
  export HF_HOME=/mnt/data/cyh/.cache/huggingface
  export TRANSFORMERS_CACHE=/mnt/data/cyh/.cache/huggingface
  export HF_HUB_DISABLE_TELEMETRY=1
  export TFDS_DATA_DIR=/mnt/data/cyh/tensorflow_datasets
  export PYTHONPATH="$ROOT/LIBERO:$ROOT:${PYTHONPATH:-}"

  stamp=$(date +%Y%m%d_%H%M%S)
  save_root="results/interp_projector/${RUN_ID}/0/${TAG}_30trials_egl_${stamp}"
  echo "save_root=${save_root}"
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true
  python vla_scripts/parallel_libero_evaluator_egl.py \
    --num-trails-per-task 30 \
    --num-gpus 2 \
    --num-processes 10 \
    --task-suite-name libero_core \
    --pretrained-checkpoint "runs/interp_projector/${RUN_ID}" \
    --unnorm_key libero_core_lt \
    --save-root "${save_root}" \
    --steps 0 \
    --instruction-formatting False

  echo "== Projector-only interp eval done $(date -Is) =="
  grep -R -E "Overall success rate|Task .*success rate" "${save_root}" 2>/dev/null || true
} >> "${LOG}" 2>&1
