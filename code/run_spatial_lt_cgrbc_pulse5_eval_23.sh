#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/data/cyh/VLA-long-tail
ENV=/mnt/data/cyh/envs/vla-long-tail
RUN_ROOT=runs/spatial_lt_cgrbc
RESULT_ROOT=results/spatial_lt_cgrbc_pulse5
SUITE=libero_spatial
UNNORM_KEY=libero_spatial_lt
NUM_TRIALS=${NUM_TRIALS:-10}
RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}
LOG=/mnt/data/cyh/spatial_lt_cgrbc_pulse5_eval_${RUN_STAMP}.log
wait_for_gpus() {
  while true; do
    used2=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 2 | tr -dc '0-9')
    used3=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3 | tr -dc '0-9')
    util2=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 2 | tr -dc '0-9')
    util3=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 3 | tr -dc '0-9')
    echo "$(date -Is) gpu2=${used2}MiB/${util2}% gpu3=${used3}MiB/${util3}%"
    if [[ "${used2:-999999}" -lt 2000 && "${used3:-999999}" -lt 2000 && "${util2:-999999}" -lt 20 && "${util3:-999999}" -lt 20 ]]; then break; fi
    sleep 900
  done
}
link_eval_steps() {
  local run_id="$1"; local step="$2"
  local ckpt_file ckpt_base suffix alias dest
  ckpt_file=$(find "${RUN_ROOT}/${run_id}/checkpoints" -maxdepth 1 -type f -name 'step-*.pt' | sort | tail -n 1)
  [[ -n "${ckpt_file}" ]] || { echo "No checkpoint for ${run_id}" >&2; return 1; }
  ckpt_base=$(basename "${ckpt_file}")
  suffix="${ckpt_base#step-}"; suffix="${suffix#*-}"
  for alias in "${step}" "$(printf "%03d" "${step}")" "$(printf "%04d" "${step}")" "$(printf "%05d" "${step}")" "$(printf "%06d" "${step}")"; do
    dest="step-${alias}-${suffix}"
    [[ "${dest}" == "${ckpt_base}" ]] || ln -sf "${ckpt_base}" "${RUN_ROOT}/${run_id}/checkpoints/${dest}"
  done
}
run_eval() {
  local seed="$1"; local run_id="$2"
  local save_root="${RESULT_ROOT}/${run_id}/step5/pulse5_seed${seed}_${SUITE}_${NUM_TRIALS}trials_egl_$(date +%Y%m%d_%H%M%S)"
  link_eval_steps "${run_id}" 5
  python vla_scripts/parallel_libero_evaluator_egl.py \
    --num-trails-per-task "${NUM_TRIALS}" \
    --num-gpus 2 \
    --num-processes 10 \
    --task-suite-name "${SUITE}" \
    --pretrained-checkpoint "${RUN_ROOT}/${run_id}" \
    --unnorm_key "${UNNORM_KEY}" \
    --save-root "${save_root}" \
    --steps 5 \
    --instruction-formatting False
  grep -R -E "Overall success rate|Task .*success rate" "${save_root}" 2>/dev/null || true
}
{
  echo "== CGRBC pulse5 eval start $(date -Is) stamp=${RUN_STAMP} =="
  source /opt/miniconda3/etc/profile.d/conda.sh
  conda activate "${ENV}"
  cd "${ROOT}"
  export CUDA_VISIBLE_DEVICES=2,3
  export HF_ENDPOINT=https://hf-mirror.com
  export HF_HOME=/mnt/data/cyh/.cache/huggingface
  export TRANSFORMERS_CACHE=/mnt/data/cyh/.cache/huggingface
  export TOKENIZERS_PARALLELISM=false
  export WANDB_DISABLED=true
  export PRISMATIC_DATA_ROOT="${ROOT}/data/prismatic"
  export PYTHONPATH="${ROOT}/LIBERO:${ROOT}:${PYTHONPATH:-}"
  export MUJOCO_GL=egl
  export PYTHONUNBUFFERED=1
  export TFDS_DATA_DIR=/mnt/data/cyh/tensorflow_datasets
  mkdir -p "${RESULT_ROOT}"
  wait_for_gpus
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
  run_eval 7 cgrbc_smoke_libero_spatial_lt_base1000p5_seed7_a10p0_20260719_030410
  run_eval 13 cgrbc_smoke_libero_spatial_lt_base1000p5_seed13_a10p0_20260719_030410
  echo "== CGRBC pulse5 eval all done $(date -Is) =="
} >> "${LOG}" 2>&1
