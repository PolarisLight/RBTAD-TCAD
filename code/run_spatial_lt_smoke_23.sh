#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/data/cyh/VLA-long-tail
ENV=/mnt/data/cyh/envs/vla-long-tail
RUN_ROOT=runs/spatial_lt_smoke
DATA_MIX=libero_spatial_lt
SUITE=libero_spatial
STEPS=${STEPS:-5}
SEED=${SEED:-7}

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
unset TARGET_TASK_INSTRUCTION
unset TARGET_TASK_WEIGHT
mkdir -p "$PRISMATIC_DATA_ROOT" "$HF_HOME" "$RUN_ROOT"
: > .hf_token

echo "== smoke gpu check $(date -Is) =="
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits

run_train() {
  local mode="$1"
  local port="$2"
  local tcad_lambda="0.0"
  local tcad_ratio="0.0"
  local tail_max_count="0"
  local conf_gate="none"
  local rare_bc_max_count="0"
  local rare_bc_weight="1.0"
  local run_id="${mode}_${DATA_MIX}_s${STEPS}_seed${SEED}_b20_$(date +%Y%m%d_%H%M%S)"

  if [[ "$mode" == "rbtad" ]]; then
    tcad_lambda="0.1"
    tcad_ratio="1.0"
    tail_max_count="9"
    conf_gate="batch_median"
    rare_bc_max_count="9"
    rare_bc_weight="2.0"
  fi

  echo "== ${mode} train start $(date -Is) run_id=${run_id} =="
  torchrun --nnodes 1 --nproc-per-node 2 --master_addr 127.0.0.1 --master_port "${port}" vla_scripts/train.py \
    --pretrained_checkpoint pretrained/minivla-libero90-prismatic/checkpoints/step-122500-epoch-55-loss=0.0743.pt \
    --vla.type "prism-qwen25-dinosiglip-224px+0_5b+mx-libero-core-lt" \
    --vla.data_mix "${DATA_MIX}" \
    --vla.expected_world_size 2 \
    --vla.global_batch_size 20 \
    --vla.per_device_batch_size 10 \
    --data_root_dir /mnt/data/cyh/tensorflow_datasets \
    --run_root_dir "${RUN_ROOT}" \
    --run_id "${run_id}" \
    --save_interval "${STEPS}" \
    --seed "${SEED}" \
    --tcad_lambda "${tcad_lambda}" \
    --tcad_ratio "${tcad_ratio}" \
    --tcad_margin 0.2 \
    --tcad_tail_max_count "${tail_max_count}" \
    --tcad_conf_gate "${conf_gate}" \
    --tcad_negative_mode manifest \
    --rare_bc_max_count "${rare_bc_max_count}" \
    --rare_bc_weight "${rare_bc_weight}" \
    --train_limit_steps "${STEPS}"
  echo "== ${mode} train done $(date -Is) =="

  local debug="${RUN_ROOT}/${run_id}/tcad-debug.csv"
  echo "debug=${debug}"
  tail -20 "${debug}" || true
  if [[ "$mode" == "rbtad" ]]; then
    awk -F, 'NR>1 {active+=$3; tail+=$5; weighted+=$6} END {printf("summary active=%d tail=%d weighted=%d\n", active, tail, weighted); if (active <= 0 || tail <= 0 || weighted <= 0) exit 9}' "${debug}"
  fi
}

run_train baseline 29701
run_train rbtad 29702

echo "== smoke all done $(date -Is) =="
