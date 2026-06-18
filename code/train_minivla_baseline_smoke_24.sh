#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/data/cyh/VLA-long-tail
ENV=/mnt/data/cyh/envs/vla-long-tail
LOG=/mnt/data/cyh/vla_train_smoke_24.log

{
  echo "== smoke start $(date -Is) =="
  source /opt/miniconda3/etc/profile.d/conda.sh
  conda activate "$ENV"
  cd "$ROOT"
  export CUDA_VISIBLE_DEVICES=0
  export HF_ENDPOINT=https://hf-mirror.com
  export TOKENIZERS_PARALLELISM=false
  export WANDB_DISABLED=true
  export PRISMATIC_DATA_ROOT="$ROOT/data/prismatic"
  export PYTHONPATH="$ROOT/LIBERO:$ROOT:${PYTHONPATH:-}"
  mkdir -p "$PRISMATIC_DATA_ROOT"
  : > .hf_token

  echo "== command =="
  echo "torchrun --nnodes 1 --nproc-per-node 1 --master_addr 127.0.0.1 --master_port 29541 vla_scripts/train.py ..."

  torchrun --nnodes 1 --nproc-per-node 1 --master_addr 127.0.0.1 --master_port 29541 vla_scripts/train.py \
    --pretrained_checkpoint pretrained/minivla-libero90-prismatic/checkpoints/step-122500-epoch-55-loss=0.0743.pt \
    --vla.type "prism-qwen25-dinosiglip-224px+0_5b+mx-libero-core-full" \
    --vla.expected_world_size 1 \
    --vla.global_batch_size 1 \
    --vla.per_device_batch_size 1 \
    --data_root_dir /mnt/data/cyh/tensorflow_datasets \
    --run_root_dir runs/smoke_baseline_full \
    --run_id smoke_full_1gpu_1step \
    --max_steps 1 \
    --save_interval 999999

  echo "== smoke done $(date -Is) =="
  find runs/smoke_baseline_full/smoke_full_1gpu_1step -maxdepth 3 -type f -printf "%p %s\n" | sort
} >> "$LOG" 2>&1
