#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/data/cyh/VLA-long-tail
ENV=/mnt/data/cyh/envs/vla-long-tail
LOG=/mnt/data/cyh/tcad_final_smoke_23.log

{
  echo "== tcad-final smoke start $(date -Is) =="
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
  mkdir -p "$PRISMATIC_DATA_ROOT" "$HF_HOME" runs/tcad_final_lt_smoke
  : > .hf_token

  torchrun --nnodes 1 --nproc-per-node 2 --master_addr 127.0.0.1 --master_port 29556 vla_scripts/train.py \
    --pretrained_checkpoint pretrained/minivla-libero90-prismatic/checkpoints/step-122500-epoch-55-loss=0.0743.pt \
    --vla.type "prism-qwen25-dinosiglip-224px+0_5b+mx-libero-core-lt" \
    --vla.expected_world_size 2 \
    --vla.global_batch_size 8 \
    --vla.per_device_batch_size 4 \
    --data_root_dir /mnt/data/cyh/tensorflow_datasets \
    --run_root_dir runs/tcad_final_lt_smoke \
    --run_id tcad_final_s5_maskpos_r2 \
    --save_interval 5 \
    --tcad_lambda 0.1 \
    --tcad_ratio 0.25 \
    --tcad_margin 0.2 \
    --tcad_smoke_steps 5

  echo "== tcad-final smoke done $(date -Is) =="
  find runs/tcad_final_lt_smoke/tcad_final_s5_maskpos_r2 -maxdepth 4 -type f -printf "%p %s\n" | sort
} >> "$LOG" 2>&1
