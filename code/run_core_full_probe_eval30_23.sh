#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/data/cyh/VLA-long-tail
ENV=/mnt/data/cyh/envs/vla-long-tail
LOG=/mnt/data/cyh/core_full_probe_eval30.log

eval_one() {
  local mode="$1"
  local run_id="${mode}_libero_core_s1000_seed7_b20"
  local run_root="runs/cross_dataset_probe/${run_id}"
  local stamp
  stamp=$(date +%Y%m%d_%H%M%S)
  local save_root="results/cross_dataset_probe/${run_id}/1000/${mode}_libero_core_30trials_egl_${stamp}"
  echo "== ${mode} 30-trial eval start $(date -Is) save_root=${save_root} =="
  python vla_scripts/parallel_libero_evaluator_egl.py \
    --num-trails-per-task 30 \
    --num-gpus 2 \
    --num-processes 10 \
    --task-suite-name libero_core \
    --pretrained-checkpoint "${run_root}" \
    --unnorm_key libero_core_full \
    --save-root "${save_root}" \
    --steps 1000 \
    --instruction-formatting False
  echo "== ${mode} 30-trial eval done $(date -Is) =="
  grep -R -E "Overall success rate|Task .*success rate" "${save_root}" 2>/dev/null || true
}

{
  echo "== core_full 30-trial checkpoint eval start $(date -Is) =="
  source /opt/miniconda3/etc/profile.d/conda.sh
  conda activate "$ENV"
  cd "$ROOT"

  export CUDA_VISIBLE_DEVICES=2,3
  export HF_ENDPOINT=https://hf-mirror.com
  export HF_HOME=/mnt/data/cyh/.cache/huggingface
  export TOKENIZERS_PARALLELISM=false
  export WANDB_DISABLED=true
  export PRISMATIC_DATA_ROOT="$ROOT/data/prismatic"
  export PYTHONPATH="$ROOT/LIBERO:$ROOT:${PYTHONPATH:-}"
  export MUJOCO_GL=egl
  export PYTHONUNBUFFERED=1
  export TRANSFORMERS_CACHE=/mnt/data/cyh/.cache/huggingface
  export HF_HUB_DISABLE_TELEMETRY=1
  export TFDS_DATA_DIR=/mnt/data/cyh/tensorflow_datasets
  mkdir -p "$PRISMATIC_DATA_ROOT" "$HF_HOME"
  : > .hf_token

  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true
  eval_one baseline
  eval_one rbtad
  echo "== core_full 30-trial checkpoint eval all done $(date -Is) =="
} >> "${LOG}" 2>&1
