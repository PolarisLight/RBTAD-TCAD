#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/data/cyh/VLA-long-tail
ENV=/mnt/data/cyh/envs/vla-long-tail
SUITE=${SUITE:-libero_spatial}
UNNORM_KEY=${UNNORM_KEY:-libero_spatial_lt}
NUM_TRIALS=${NUM_TRIALS:-30}
RESULT_ROOT=results/spatial_lt_confirm30
LOG=/mnt/data/cyh/spatial_lt_confirm30_rsdf_vs_baseline.log

run_eval() {
  local label="$1"
  local ckpt_dir="$2"
  local step="$3"
  local stamp save_root
  stamp=$(date +%Y%m%d_%H%M%S)
  save_root="${RESULT_ROOT}/${label}/step${step}/${label}_${SUITE}_${NUM_TRIALS}trials_egl_${stamp}"
  echo "== eval start $(date -Is) label=${label} ckpt=${ckpt_dir} step=${step} save_root=${save_root} =="
  python vla_scripts/parallel_libero_evaluator_egl.py \
    --num-trails-per-task "${NUM_TRIALS}" \
    --num-gpus 2 \
    --num-processes 10 \
    --task-suite-name "${SUITE}" \
    --pretrained-checkpoint "${ckpt_dir}" \
    --unnorm_key "${UNNORM_KEY}" \
    --save-root "${save_root}" \
    --steps "${step}" \
    --instruction-formatting False
  echo "== eval done $(date -Is) label=${label} =="
  grep -R -E "Overall success rate|Task .*success rate" "${save_root}" 2>/dev/null || true
}

{
  echo "== confirm30 start $(date -Is) =="
  source /opt/miniconda3/etc/profile.d/conda.sh
  conda activate "$ENV"
  cd "$ROOT"
  export CUDA_VISIBLE_DEVICES=2,3
  export HF_ENDPOINT=https://hf-mirror.com
  export HF_HOME=/mnt/data/cyh/.cache/huggingface
  export TRANSFORMERS_CACHE=/mnt/data/cyh/.cache/huggingface
  export TOKENIZERS_PARALLELISM=false
  export WANDB_DISABLED=true
  export PRISMATIC_DATA_ROOT="$ROOT/data/prismatic"
  export PYTHONPATH="$ROOT/LIBERO:$ROOT:${PYTHONPATH:-}"
  export MUJOCO_GL=egl
  export PYTHONUNBUFFERED=1
  export TFDS_DATA_DIR=/mnt/data/cyh/tensorflow_datasets
  mkdir -p "$RESULT_ROOT" "$PRISMATIC_DATA_ROOT" "$HF_HOME"
  : > .hf_token
  echo "== gpu precheck $(date -Is) =="
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
  used2=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 2 | tr -dc '0-9')
  used3=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3 | tr -dc '0-9')
  if [[ "${used2:-999999}" -gt 2000 || "${used3:-999999}" -gt 2000 ]]; then
    echo "GPU 2/3 not free enough: gpu2=${used2}MiB gpu3=${used3}MiB"
    exit 3
  fi
  run_eval "rsdf_visionllm_barc100_a0p5" "runs/spatial_lt_selective_soup/rsdf_barc100_visionllm_a0p5" 100
  run_eval "baseline_libero_spatial_lt_s1000_seed7_b20" "runs/spatial_lt_screen/baseline_libero_spatial_lt_s1000_seed7_b20" 1000
  echo "== confirm30 all done $(date -Is) =="
} >> "$LOG" 2>&1