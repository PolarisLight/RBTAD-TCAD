#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/data/cyh/VLA-long-tail
ENV=/mnt/data/cyh/envs/vla-long-tail
SUITE=${SUITE:-libero_spatial}
UNNORM_KEY=${UNNORM_KEY:-libero_spatial_lt}
NUM_TRIALS=${NUM_TRIALS:-30}
RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}
RESULT_ROOT=results/spatial_lt_bpc_rsdf_confirm30
LOG=/mnt/data/cyh/spatial_lt_bpc_rsdf_confirm30_${RUN_STAMP}.log

BASE7=runs/spatial_lt_screen/baseline_libero_spatial_lt_s1000_seed7_b20
RSDF7=runs/spatial_lt_selective_soup/rsdf_barc100_visionllm_a0p5
BPC7=runs/spatial_lt_bpc_rsdf/bpc_rsdf_libero_spatial_lt_seed7_p50_20260719_223327
BASE13=runs/spatial_lt_multiseed/baseline_libero_spatial_lt_s1000_seed13_b20_20260718_185539
RSDF13=runs/spatial_lt_rsdf_multiseed/rsdf_visionllm_barc100_seed13_a0p5_20260718_185539
BPC13=runs/spatial_lt_bpc_rsdf/bpc_rsdf_libero_spatial_lt_seed13_p50_20260719_223327

wait_for_gpus() {
  echo "== waiting for GPUs 2/3 $(date -Is) =="
  while true; do
    used2=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 2 | tr -dc '0-9')
    used3=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3 | tr -dc '0-9')
    util2=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 2 | tr -dc '0-9')
    util3=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 3 | tr -dc '0-9')
    echo "$(date -Is) gpu2=${used2}MiB/${util2}% gpu3=${used3}MiB/${util3}%"
    if [[ "${used2:-999999}" -lt 2000 && "${used3:-999999}" -lt 2000 && "${util2:-999999}" -lt 20 && "${util3:-999999}" -lt 20 ]]; then
      break
    fi
    sleep 900
  done
}

run_eval() {
  local label="$1" checkpoint="$2" step="$3"
  local save_root="${RESULT_ROOT}/${label}/step${step}/${label}_${SUITE}_${NUM_TRIALS}trials_egl_${RUN_STAMP}"
  echo "== eval start $(date -Is) label=${label} save_root=${save_root} =="
  export CUDA_VISIBLE_DEVICES=2,3
  export EVAL_ALLOWED_GPUS=2,3
  unset EVAL_INIT_IDS || true
  python vla_scripts/parallel_libero_evaluator_egl.py \
    --num-trails-per-task "${NUM_TRIALS}" \
    --num-gpus 2 \
    --num-processes 10 \
    --task-suite-name "${SUITE}" \
    --pretrained-checkpoint "${checkpoint}" \
    --unnorm_key "${UNNORM_KEY}" \
    --save-root "${save_root}" \
    --steps "${step}" \
    --instruction-formatting False
  echo "== eval done $(date -Is) label=${label} =="
  grep -R -E "Overall success rate|Task .*success rate|Init ids" "${save_root}" 2>/dev/null || true
}

{
  echo "== BPC-RSDF confirm30 start $(date -Is) stamp=${RUN_STAMP} =="
  source /opt/miniconda3/etc/profile.d/conda.sh
  conda activate "${ENV}"
  cd "${ROOT}"
  export CUDA_VISIBLE_DEVICES=2,3
  export EVAL_ALLOWED_GPUS=2,3
  export HF_ENDPOINT=https://hf-mirror.com
  export HF_HOME=/mnt/data/cyh/.cache/huggingface
  export TRANSFORMERS_CACHE=/mnt/data/cyh/.cache/huggingface
  export TOKENIZERS_PARALLELISM=false
  export WANDB_DISABLED=true
  export PRISMATIC_DATA_ROOT="${ROOT}/data/prismatic"
  export PYTHONPATH="${ROOT}/LIBERO:${ROOT}:${PYTHONPATH:-}"
  export MUJOCO_GL=egl
  export PYTHONUNBUFFERED=1
  export HF_HUB_DISABLE_TELEMETRY=1
  export TFDS_DATA_DIR=/mnt/data/cyh/tensorflow_datasets
  mkdir -p "${PRISMATIC_DATA_ROOT}" "${HF_HOME}" "${RESULT_ROOT}"
  : > .hf_token
  wait_for_gpus
  python /mnt/data/cyh/patch_eval_fixed_init_ids.py

  run_eval baseline_seed7 "${BASE7}" 1000
  run_eval rsdf_seed7 "${RSDF7}" 100
  run_eval bpc_rsdf_seed7 "${BPC7}" 50
  run_eval baseline_seed13 "${BASE13}" 1000
  run_eval rsdf_seed13 "${RSDF13}" 100
  run_eval bpc_rsdf_seed13 "${BPC13}" 50
  echo "== BPC-RSDF confirm30 all done $(date -Is) =="
} >> "${LOG}" 2>&1