#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/data/cyh/VLA-long-tail
ENV=/mnt/data/cyh/envs/vla-long-tail
LOG=/mnt/data/cyh/rbtad_full_23.log

{
  echo "== RBTAD full start $(date -Is) =="
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
  mkdir -p "$PRISMATIC_DATA_ROOT" "$HF_HOME" runs/rbtad_lt_main
  : > .hf_token

  echo "== gpu snapshot =="
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
  echo "== command =="
  echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES torchrun --nproc-per-node 2 ... rbtad_w3_tail9_confmedian_seed7_b20"

  torchrun --nnodes 1 --nproc-per-node 2 --master_addr 127.0.0.1 --master_port 29564 vla_scripts/train.py \
    --pretrained_checkpoint pretrained/minivla-libero90-prismatic/checkpoints/step-122500-epoch-55-loss=0.0743.pt \
    --vla.type "prism-qwen25-dinosiglip-224px+0_5b+mx-libero-core-lt" \
    --vla.expected_world_size 2 \
    --vla.global_batch_size 20 \
    --vla.per_device_batch_size 10 \
    --data_root_dir /mnt/data/cyh/tensorflow_datasets \
    --run_root_dir runs/rbtad_lt_main \
    --run_id rbtad_w3_tail9_confmedian_seed7_b20 \
    --save_interval 5000 \
    --tcad_lambda 0.1 \
    --tcad_ratio 0.5 \
    --tcad_margin 0.2 \
    --tcad_tail_max_count 9 \
    --tcad_conf_gate batch_median \
    --rare_bc_max_count 9 \
    --rare_bc_weight 3.0

  echo "== RBTAD full done $(date -Is) =="
  find runs/rbtad_lt_main/rbtad_w3_tail9_confmedian_seed7_b20 -maxdepth 4 -type f -printf "%p %s\n" | sort
} >> "$LOG" 2>&1
