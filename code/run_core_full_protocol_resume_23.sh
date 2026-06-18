#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/data/cyh/VLA-long-tail
ENV=/mnt/data/cyh/envs/vla-long-tail
RUN_ROOT=runs/core_full_protocol
RESULT_ROOT=results/core_full_protocol
LOG=/mnt/data/cyh/core_full_protocol_resume_23.log
MIN_FULL_STEP=100000

latest_real_step_for_run() {
  local run_id="$1"
  find "${RUN_ROOT}/${run_id}/checkpoints" -maxdepth 1 -type f -name 'step-*.pt' \
    | sed -E 's/.*step-([0-9]+)-.*/\1/' \
    | sort -n \
    | tail -1 \
    | sed -E 's/^0+([0-9])/\1/'
}

eval_one() {
  local mode="$1"
  local run_id="$2"
  local step="$3"
  local stamp
  stamp=$(date +%Y%m%d_%H%M%S)
  local save_root="${RESULT_ROOT}/${run_id}/${step}/${mode}_libero_core_full_50trials_egl_${stamp}"

  echo "== ${mode} protocol eval start $(date -Is) step=${step} save_root=${save_root} =="
  python vla_scripts/parallel_libero_evaluator_egl.py \
    --num-trails-per-task 50 \
    --num-gpus 2 \
    --num-processes 10 \
    --task-suite-name libero_core \
    --pretrained-checkpoint "${RUN_ROOT}/${run_id}" \
    --unnorm_key libero_core_full \
    --save-root "${save_root}" \
    --steps "${step}" \
    --instruction-formatting False
  echo "== ${mode} protocol eval done $(date -Is) =="
  grep -R -E "Overall success rate|Task .*success rate" "${save_root}" 2>/dev/null || true
}

train_rbtad() {
  local run_id=rbtad_libero_core_full_alltcad_protocol_seed7_b20
  local latest
  latest=$(latest_real_step_for_run "${run_id}" 2>/dev/null || true)
  if [[ -n "${latest}" ]] && (( latest >= MIN_FULL_STEP )); then
    echo "== rbtad full train already has step ${latest}; skipping train =="
    echo "${latest}" > "${RUN_ROOT}/${run_id}/selected_eval_step.txt"
    return 0
  fi

  echo "== rbtad full train start $(date -Is) run_id=${run_id} =="
  torchrun --nnodes 1 --nproc-per-node 2 --master_addr 127.0.0.1 --master_port 29632 vla_scripts/train.py \
    --pretrained_checkpoint pretrained/minivla-libero90-prismatic/checkpoints/step-122500-epoch-55-loss=0.0743.pt \
    --vla.type "prism-qwen25-dinosiglip-224px+0_5b+mx-libero-core-lt" \
    --vla.data_mix libero_core_full \
    --vla.expected_world_size 2 \
    --vla.global_batch_size 20 \
    --vla.per_device_batch_size 10 \
    --data_root_dir /mnt/data/cyh/tensorflow_datasets \
    --run_root_dir "${RUN_ROOT}" \
    --run_id "${run_id}" \
    --save_interval 5000 \
    --tcad_lambda 0.1 \
    --tcad_ratio 0.5 \
    --tcad_margin 0.2 \
    --tcad_tail_max_count 1000000000 \
    --tcad_conf_gate batch_median \
    --rare_bc_max_count 0 \
    --rare_bc_weight 1.0
  echo "== rbtad full train exit $(date -Is) status=0 =="

  latest=$(latest_real_step_for_run "${run_id}")
  if [[ -z "${latest}" ]] || (( latest < MIN_FULL_STEP )); then
    echo "latest RBTAD checkpoint step ${latest:-none} is below ${MIN_FULL_STEP}; aborting eval"
    exit 1
  fi
  echo "${latest}" > "${RUN_ROOT}/${run_id}/selected_eval_step.txt"
}

{
  echo "== core_full protocol resume start $(date -Is) =="
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

  mkdir -p "$PRISMATIC_DATA_ROOT" "$HF_HOME" "$RUN_ROOT" "$RESULT_ROOT"
  : > .hf_token

  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true

  baseline_run=baseline_libero_core_full_protocol_seed7_b20
  baseline_step=$(latest_real_step_for_run "${baseline_run}")
  if (( baseline_step < MIN_FULL_STEP )); then
    echo "baseline real checkpoint step ${baseline_step} is below ${MIN_FULL_STEP}; aborting"
    exit 1
  fi
  echo "${baseline_step}" > "${RUN_ROOT}/${baseline_run}/selected_eval_step.txt"
  eval_one baseline "${baseline_run}" "${baseline_step}"

  train_rbtad
  rbtad_run=rbtad_libero_core_full_alltcad_protocol_seed7_b20
  rbtad_step=$(cat "${RUN_ROOT}/${rbtad_run}/selected_eval_step.txt")
  eval_one rbtad "${rbtad_run}" "${rbtad_step}"
  echo "== core_full protocol resume all done $(date -Is) =="
} >> "${LOG}" 2>&1
