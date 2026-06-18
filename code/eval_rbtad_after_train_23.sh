#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/data/cyh/VLA-long-tail
ENV=/mnt/data/cyh/envs/vla-long-tail
RUN_ID=rbtad_w3_tail9_confmedian_seed7_b20
RUN_ROOT=runs/rbtad_lt_main/$RUN_ID
LOG=/mnt/data/cyh/rbtad_eval_after_train_23.log

{
  echo "== RBTAD eval watcher start $(date -Is) =="
  while pgrep -af "train_rbtad_full_23|vla_scripts/train.py" >/dev/null; do
    echo "waiting for RBTAD training $(date -Is)"
    sleep 1800
  done

  echo "== training no longer running $(date -Is) =="
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
  mkdir -p "$HF_HOME" "$PRISMATIC_DATA_ROOT"
  : > .hf_token

  latest_ckpt=$(find "$RUN_ROOT/checkpoints" -maxdepth 1 -type f -name 'step-*.pt' | sed -E 's/.*step-([0-9]+)-.*/\1/' | sort -n | tail -1)
  if [[ -z "${latest_ckpt:-}" ]]; then
    echo "no checkpoint found under $RUN_ROOT/checkpoints"
    exit 2
  fi
  stamp=$(date +%Y%m%d_%H%M%S)
  save_root="results/rbtad_lt_main/$RUN_ID/$latest_ckpt/rbtad_step${latest_ckpt}_30trials_egl_${stamp}"

  echo "== eval start $(date -Is) step=$latest_ckpt save_root=$save_root =="
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true
  python vla_scripts/parallel_libero_evaluator_egl.py \
    --num-trails-per-task 30 \
    --num-gpus 2 \
    --num-processes 10 \
    --task-suite-name libero_core \
    --pretrained-checkpoint "$RUN_ROOT" \
    --unnorm_key libero_core_lt \
    --save-root "$save_root" \
    --steps "$latest_ckpt" \
    --instruction-formatting False

  echo "== eval done $(date -Is) =="
  grep -R -E "Overall success rate|Task .*success rate" "$save_root" 2>/dev/null || true
} >> "$LOG" 2>&1
