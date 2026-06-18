#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/data/cyh/VLA-long-tail
ENV=/mnt/data/cyh/envs/vla-long-tail
LOG=/mnt/data/cyh/vla_train_baseline_lt_full_23.log

{
  echo "== baseline lt full start $(date -Is) =="
  source /opt/miniconda3/etc/profile.d/conda.sh
  conda activate "$ENV"
  cd "$ROOT"

  export CUDA_VISIBLE_DEVICES=0,1,2,3
  export HF_ENDPOINT=https://hf-mirror.com
  export HF_HOME=/mnt/data/cyh/.cache/huggingface
  export TOKENIZERS_PARALLELISM=false
  export WANDB_DISABLED=true
  export PRISMATIC_DATA_ROOT="$ROOT/data/prismatic"
  export PYTHONPATH="$ROOT/LIBERO:$ROOT:${PYTHONPATH:-}"
  mkdir -p "$PRISMATIC_DATA_ROOT" "$HF_HOME" runs/miniVLA_libero_core_lt
  : > .hf_token

  echo "== command =="
  echo "torchrun --nnodes 1 --nproc-per-node 4 --master_addr 127.0.0.1 --master_port 29543 vla_scripts/train.py ..."

  torchrun --nnodes 1 --nproc-per-node 4 --master_addr 127.0.0.1 --master_port 29543 vla_scripts/train.py \
    --pretrained_checkpoint pretrained/minivla-libero90-prismatic/checkpoints/step-122500-epoch-55-loss=0.0743.pt \
    --vla.type "prism-qwen25-dinosiglip-224px+0_5b+mx-libero-core-lt" \
    --data_root_dir /mnt/data/cyh/tensorflow_datasets \
    --run_root_dir runs/miniVLA_libero_core_lt

  echo "== baseline lt full done $(date -Is) =="
  find runs/miniVLA_libero_core_lt -maxdepth 4 -type f -printf "%p %s\n" | sort
} >> "$LOG" 2>&1
