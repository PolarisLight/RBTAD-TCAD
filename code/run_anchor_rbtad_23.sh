#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/data/cyh/VLA-long-tail
ENV=/mnt/data/cyh/envs/vla-long-tail
LIMIT_STEPS=${LIMIT_STEPS:-500}
RUN_ID=${RUN_ID:-anchor_rbtad_l2head_w2_tail9_s${LIMIT_STEPS}_seed7_b20}
RUN_ROOT=runs/anchor_rbtad/${RUN_ID}
LOG=/mnt/data/cyh/${RUN_ID}.log
DO_EVAL=${DO_EVAL:-0}

{
  echo "== Anchor-RBTAD start $(date -Is) =="
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
  mkdir -p "$PRISMATIC_DATA_ROOT" "$HF_HOME" runs/anchor_rbtad
  : > .hf_token

  echo "run_id=${RUN_ID}"
  echo "limit_steps=${LIMIT_STEPS}"
  echo "do_eval=${DO_EVAL}"
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true

  torchrun --nnodes 1 --nproc-per-node 2 --master_addr 127.0.0.1 --master_port 29581 vla_scripts/train.py \
    --pretrained_checkpoint runs/rbtad_lt_main/rbtad_w3_tail9_confmedian_seed7_b20/checkpoints/step-034075-epoch-36-loss=0.0678.pt \
    --vla.type "prism-qwen25-dinosiglip-224px+0_5b+mx-libero-core-lt" \
    --vla.expected_world_size 2 \
    --vla.global_batch_size 20 \
    --vla.per_device_batch_size 10 \
    --data_root_dir /mnt/data/cyh/tensorflow_datasets \
    --run_root_dir runs/anchor_rbtad \
    --run_id "${RUN_ID}" \
    --save_interval 1000 \
    --tcad_lambda 0.0 \
    --tcad_ratio 0.0 \
    --tcad_margin 0.2 \
    --tcad_tail_max_count 0 \
    --tcad_conf_gate none \
    --rare_bc_max_count 9 \
    --rare_bc_weight 1.25 \
    --anchor_l2_lambda 0.05 \
    --anchor_l2_filter "" \
    --train_limit_steps "${LIMIT_STEPS}"

  echo "== Anchor-RBTAD train done $(date -Is) =="
  find "${RUN_ROOT}/checkpoints" -maxdepth 1 -type f -name 'step-*.pt' -printf "%f %s\n" | sort || true
  cat "${RUN_ROOT}/tcad-debug.csv" 2>/dev/null | tail -10 || true

  if [[ "${DO_EVAL}" != "1" ]]; then
    echo "== eval skipped =="
    exit 0
  fi

  latest_ckpt=$(find "${RUN_ROOT}/checkpoints" -maxdepth 1 -type f -name 'step-*.pt' | sed -E 's/.*step-([0-9]+)-.*/\1/' | sort -n | tail -1)
  if [[ -z "${latest_ckpt:-}" ]]; then
    echo "no checkpoint found under ${RUN_ROOT}/checkpoints"
    exit 2
  fi
  eval_step=$((10#${latest_ckpt}))
  ckpt_file=$(find "${RUN_ROOT}/checkpoints" -maxdepth 1 -type f -name "step-${latest_ckpt}-*.pt" | sort | tail -1)
  ckpt_base=$(basename "${ckpt_file}")
  eval_ckpt_base=$(echo "${ckpt_base}" | sed -E "s/step-${latest_ckpt}-/step-${eval_step}-/")
  if [[ ! -e "${RUN_ROOT}/checkpoints/${eval_ckpt_base}" ]]; then
    ln -s "${ckpt_base}" "${RUN_ROOT}/checkpoints/${eval_ckpt_base}"
  fi
  stamp=$(date +%Y%m%d_%H%M%S)
  save_root="results/anchor_rbtad/${RUN_ID}/${eval_step}/anchor_step${eval_step}_30trials_egl_${stamp}"

  echo "== Anchor-RBTAD eval start $(date -Is) step=${eval_step} save_root=${save_root} =="
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

  echo "== Anchor-RBTAD eval done $(date -Is) =="
  grep -R -E "Overall success rate|Task .*success rate" "${save_root}" 2>/dev/null || true
} >> "${LOG}" 2>&1
