#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/data/cyh/VLA-long-tail
ENV=/mnt/data/cyh/envs/vla-long-tail
RUN_ID=rctad_tail9_confmedian_ratio05_seed7_b20
RUN_ROOT=runs/rctad_lt_main/$RUN_ID
LOG=/mnt/data/cyh/rctad_ckpt_sweep_23.log
STEPS=(15000 20000 25000 30000)

{
  echo "== RCTAD checkpoint sweep start $(date -Is) =="
  source /opt/miniconda3/etc/profile.d/conda.sh
  conda activate "$ENV"
  cd "$ROOT"

  export CUDA_VISIBLE_DEVICES=1,3
  export HF_ENDPOINT=https://hf-mirror.com
  export HF_HOME=/mnt/data/cyh/.cache/huggingface
  export TOKENIZERS_PARALLELISM=false
  export WANDB_DISABLED=true
  export PRISMATIC_DATA_ROOT="$ROOT/data/prismatic"
  export PYTHONPATH="$ROOT/LIBERO:$ROOT:${PYTHONPATH:-}"
  mkdir -p "$HF_HOME" "$PRISMATIC_DATA_ROOT"
  : > .hf_token

  for step in "${STEPS[@]}"; do
    if ! find "$RUN_ROOT/checkpoints" -maxdepth 1 -type f -name "step-$(printf '%06d' "$step")-*.pt" | grep -q .; then
      echo "missing checkpoint step=$step"
      continue
    fi
    stamp=$(date +%Y%m%d_%H%M%S)
    save_root="results/rctad_lt_main/$RUN_ID/$step/rctad_step${step}_30trials_egl_${stamp}"
    echo "== eval start $(date -Is) step=$step save_root=$save_root =="
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true
    python vla_scripts/parallel_libero_evaluator_egl.py \
      --num-trails-per-task 30 \
      --num-gpus 2 \
      --num-processes 10 \
      --task-suite-name libero_core \
      --pretrained-checkpoint "$RUN_ROOT" \
      --unnorm_key libero_core_lt \
      --save-root "$save_root" \
      --steps "$step" \
      --instruction-formatting False
    echo "== eval done $(date -Is) step=$step =="
    grep -R -E "Overall success rate|Task .*success rate" "$save_root" 2>/dev/null || true
  done
  echo "== RCTAD checkpoint sweep done $(date -Is) =="
} >> "$LOG" 2>&1
