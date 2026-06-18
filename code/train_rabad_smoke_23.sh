#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/data/cyh/VLA-long-tail
ENV=/mnt/data/cyh/envs/vla-long-tail
LOG=/mnt/data/cyh/rabad_smoke_23.log

{
  echo "== RABAD smoke start $(date -Is) =="
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
  mkdir -p "$PRISMATIC_DATA_ROOT" "$HF_HOME" runs/rabad_lt_smoke
  : > .hf_token

  torchrun --nnodes 1 --nproc-per-node 2 --master_addr 127.0.0.1 --master_port 29574 vla_scripts/train.py \
    --pretrained_checkpoint pretrained/minivla-libero90-prismatic/checkpoints/step-122500-epoch-55-loss=0.0743.pt \
    --vla.type "prism-qwen25-dinosiglip-224px+0_5b+mx-libero-core-lt" \
    --vla.expected_world_size 2 \
    --vla.global_batch_size 20 \
    --vla.per_device_batch_size 10 \
    --data_root_dir /mnt/data/cyh/tensorflow_datasets \
    --run_root_dir runs/rabad_lt_smoke \
    --run_id rabad_relation_anchor_w3_tail9_s5_seed7 \
    --save_interval 5000 \
    --tcad_lambda 0.1 \
    --tcad_ratio 0.5 \
    --tcad_margin 0.2 \
    --tcad_tail_max_count 9 \
    --tcad_conf_gate batch_median \
    --tcad_negative_mode relation_anchor \
    --rare_bc_max_count 9 \
    --rare_bc_weight 3.0 \
    --tcad_smoke_steps 5

  echo "== RABAD smoke done $(date -Is) =="
  cat runs/rabad_lt_smoke/rabad_relation_anchor_w3_tail9_s5_seed7/tcad-debug.csv || true
} >> "$LOG" 2>&1
