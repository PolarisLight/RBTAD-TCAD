#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/data/cyh/VLA-long-tail
ENV=/mnt/data/cyh/envs/vla-long-tail
LIMIT_STEPS=${LIMIT_STEPS:-100}
TARGET_WEIGHT=${TARGET_WEIGHT:-8.0}
RUN_ID=${RUN_ID:-balanced_tail_projector_t7t9_w${TARGET_WEIGHT}_s${LIMIT_STEPS}_seed7_b20}
RUN_ROOT=runs/balanced_tail_projector/${RUN_ID}
LOG=/mnt/data/cyh/${RUN_ID}.log
DO_EVAL=${DO_EVAL:-0}

{
  echo "== Balanced-tail projector relation start $(date -Is) =="
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
  export TARGET_TASK_INSTRUCTION="put the cream cheese in the bowl||put the wine bottle on the rack"
  export TARGET_TASK_WEIGHT="${TARGET_WEIGHT}"
  mkdir -p "$PRISMATIC_DATA_ROOT" "$HF_HOME" runs/balanced_tail_projector
  : > .hf_token

  echo "run_id=${RUN_ID}"
  echo "limit_steps=${LIMIT_STEPS}"
  echo "target_task_instruction=${TARGET_TASK_INSTRUCTION}"
  echo "target_task_weight=${TARGET_TASK_WEIGHT}"
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true

  set +e
  torchrun --nnodes 1 --nproc-per-node 2 --master_addr 127.0.0.1 --master_port 29595 vla_scripts/train.py \
    --pretrained_checkpoint runs/rbtad_lt_main/rbtad_w3_tail9_confmedian_seed7_b20/checkpoints/step-034075-epoch-36-loss=0.0678.pt \
    --vla.type "prism-qwen25-dinosiglip-224px+0_5b+mx-libero-core-lt" \
    --vla.expected_world_size 2 \
    --vla.global_batch_size 20 \
    --vla.per_device_batch_size 10 \
    --data_root_dir /mnt/data/cyh/tensorflow_datasets \
    --run_root_dir runs/balanced_tail_projector \
    --run_id "${RUN_ID}" \
    --save_interval 1000 \
    --tcad_lambda 0.03 \
    --tcad_ratio 1.0 \
    --tcad_margin 0.2 \
    --tcad_tail_max_count 6 \
    --tcad_conf_gate batch_median \
    --tcad_negative_mode relation_anchor \
    --rare_bc_max_count 0 \
    --rare_bc_weight 1.0 \
    --anchor_l2_lambda 0.05 \
    --anchor_l2_filter "projector" \
    --trainable_filter "projector" \
    --train_limit_steps "${LIMIT_STEPS}"
  train_status=$?
  set -e

  echo "== Balanced-tail projector relation train done $(date -Is) =="
  find "${RUN_ROOT}/checkpoints" -maxdepth 1 -type f -name 'step-*.pt' -printf "%f %s\n" | sort || true
  cat "${RUN_ROOT}/tcad-debug.csv" 2>/dev/null | tail -20 || true
  if [[ "${train_status}" != "0" ]]; then
    if find "${RUN_ROOT}/checkpoints" -maxdepth 1 -type f -name 'step-*.pt' | grep -q .; then
      echo "torchrun exited with status ${train_status} after checkpoint save; continuing"
    else
      echo "torchrun failed with status ${train_status} and no checkpoint was saved"
      exit "${train_status}"
    fi
  fi

  if [[ "${DO_EVAL}" != "1" ]]; then
    echo "== eval skipped =="
    exit 0
  fi

  latest_ckpt=$(find "${RUN_ROOT}/checkpoints" -maxdepth 1 -type f -name 'step-*.pt' | sed -E 's/.*step-([0-9]+)-.*/\1/' | sort -n | tail -1)
  eval_step=$((10#${latest_ckpt}))
  ckpt_file=$(find "${RUN_ROOT}/checkpoints" -maxdepth 1 -type f -name "step-${latest_ckpt}-*.pt" | sort | tail -1)
  ckpt_base=$(basename "${ckpt_file}")
  padded=$(printf "%05d" "${eval_step}")
  ln -sf "${ckpt_base}" "${RUN_ROOT}/checkpoints/step-${eval_step}-${ckpt_base#step-${latest_ckpt}-}"
  ln -sf "${ckpt_base}" "${RUN_ROOT}/checkpoints/step-${padded}-${ckpt_base#step-${latest_ckpt}-}"

  unset CUDA_VISIBLE_DEVICES
  export MUJOCO_GL=egl
  export PYTHONUNBUFFERED=1
  export TRANSFORMERS_CACHE=/mnt/data/cyh/.cache/huggingface
  export HF_HUB_DISABLE_TELEMETRY=1
  export TFDS_DATA_DIR=/mnt/data/cyh/tensorflow_datasets

  stamp=$(date +%Y%m%d_%H%M%S)
  save_root="results/balanced_tail_projector/${RUN_ID}/${eval_step}/balanced_tail_30trials_egl_${stamp}"
  echo "save_root=${save_root}"
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true
  python vla_scripts/parallel_libero_evaluator_egl.py \
    --num-trails-per-task 30 \
    --num-gpus 2 \
    --num-processes 10 \
    --task-suite-name libero_core \
    --pretrained-checkpoint "${RUN_ROOT}" \
    --unnorm_key libero_core_lt \
    --save-root "${save_root}" \
    --steps "${eval_step}" \
    --instruction-formatting False

  echo "== Balanced-tail projector relation eval done $(date -Is) =="
  grep -R -E "Overall success rate|Task .*success rate" "${save_root}" 2>/dev/null || true
} >> "${LOG}" 2>&1
