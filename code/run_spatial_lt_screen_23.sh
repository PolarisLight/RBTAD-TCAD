#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/data/cyh/VLA-long-tail
ENV=/mnt/data/cyh/envs/vla-long-tail
LIMIT_STEPS=${LIMIT_STEPS:-1000}
NUM_TRIALS=${NUM_TRIALS:-10}
SUITE=${SUITE:-libero_spatial}
DATA_MIX=${DATA_MIX:-libero_spatial_lt}
UNNORM_KEY=${UNNORM_KEY:-libero_spatial_lt}
SEED=${SEED:-7}
PROBE_NAME=${PROBE_NAME:-spatial_lt_matched_screen}
LOG=/mnt/data/cyh/${PROBE_NAME}_s${LIMIT_STEPS}_t${NUM_TRIALS}.log

run_one() {
  local mode="$1"
  local port="$2"
  local run_root="runs/spatial_lt_screen"
  local run_id="${mode}_${DATA_MIX}_s${LIMIT_STEPS}_seed${SEED}_b20"
  local tcad_lambda="0.0"
  local tcad_ratio="0.0"
  local rare_bc_max_count="0"
  local rare_bc_weight="1.0"
  local tail_max_count="0"
  local conf_gate="none"

  if [[ "${mode}" == "rbtad" ]]; then
    tcad_lambda="0.1"
    tcad_ratio="0.5"
    tail_max_count="9"
    conf_gate="batch_median"
    rare_bc_max_count="9"
    rare_bc_weight="2.0"
  fi

  echo "== ${mode} train start $(date -Is) =="
  echo "run_id=${run_id} suite=${SUITE} data_mix=${DATA_MIX} steps=${LIMIT_STEPS} trials=${NUM_TRIALS}"
  torchrun --nnodes 1 --nproc-per-node 2 --master_addr 127.0.0.1 --master_port "${port}" vla_scripts/train.py \
    --pretrained_checkpoint pretrained/minivla-libero90-prismatic/checkpoints/step-122500-epoch-55-loss=0.0743.pt \
    --vla.type "prism-qwen25-dinosiglip-224px+0_5b+mx-libero-core-lt" \
    --vla.data_mix "${DATA_MIX}" \
    --vla.expected_world_size 2 \
    --vla.global_batch_size 20 \
    --vla.per_device_batch_size 10 \
    --data_root_dir /mnt/data/cyh/tensorflow_datasets \
    --run_root_dir "${run_root}" \
    --run_id "${run_id}" \
    --save_interval "${LIMIT_STEPS}" \
    --seed "${SEED}" \
    --tcad_lambda "${tcad_lambda}" \
    --tcad_ratio "${tcad_ratio}" \
    --tcad_margin 0.2 \
    --tcad_tail_max_count "${tail_max_count}" \
    --tcad_conf_gate "${conf_gate}" \
    --tcad_negative_mode manifest \
    --rare_bc_max_count "${rare_bc_max_count}" \
    --rare_bc_weight "${rare_bc_weight}" \
    --train_limit_steps "${LIMIT_STEPS}"

  echo "== ${mode} train done $(date -Is) =="
  cat "${run_root}/${run_id}/tcad-debug.csv" 2>/dev/null | tail -20 || true

  local latest_ckpt
  latest_ckpt=$(find "${run_root}/${run_id}/checkpoints" -maxdepth 1 -type f -name 'step-*.pt' | sed -E 's/.*step-([0-9]+)-.*/\1/' | sort -n | tail -1)
  local eval_step=$((10#${latest_ckpt}))
  local ckpt_file
  ckpt_file=$(find "${run_root}/${run_id}/checkpoints" -maxdepth 1 -type f -name "step-${latest_ckpt}-*.pt" | sort | tail -1)
  local ckpt_base
  ckpt_base=$(basename "${ckpt_file}")
  local padded
  padded=$(printf "%05d" "${eval_step}")
  ln -sf "${ckpt_base}" "${run_root}/${run_id}/checkpoints/step-${eval_step}-${ckpt_base#step-${latest_ckpt}-}"
  ln -sf "${ckpt_base}" "${run_root}/${run_id}/checkpoints/step-${padded}-${ckpt_base#step-${latest_ckpt}-}"

  echo "== ${mode} eval start $(date -Is) =="
  local stamp
  stamp=$(date +%Y%m%d_%H%M%S)
  local save_root="results/spatial_lt_screen/${run_id}/${eval_step}/${mode}_${SUITE}_${NUM_TRIALS}trials_egl_${stamp}"
  echo "save_root=${save_root}"
  python vla_scripts/parallel_libero_evaluator_egl.py \
    --num-trails-per-task "${NUM_TRIALS}" \
    --num-gpus 2 \
    --num-processes 10 \
    --task-suite-name "${SUITE}" \
    --pretrained-checkpoint "${run_root}/${run_id}" \
    --unnorm_key "${UNNORM_KEY}" \
    --save-root "${save_root}" \
    --steps "${eval_step}" \
    --instruction-formatting False

  echo "== ${mode} eval done $(date -Is) =="
  grep -R -E "Overall success rate|Task .*success rate" "${save_root}" 2>/dev/null || true
}

{
  echo "== ${PROBE_NAME} start $(date -Is) =="
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
  mkdir -p "$PRISMATIC_DATA_ROOT" "$HF_HOME" runs/spatial_lt_screen results/spatial_lt_screen
  : > .hf_token

  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true
  run_one baseline 29801
  run_one rbtad 29802
  echo "== ${PROBE_NAME} all done $(date -Is) =="
} >> "${LOG}" 2>&1
