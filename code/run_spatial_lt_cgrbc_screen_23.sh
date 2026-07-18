#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/data/cyh/VLA-long-tail
ENV=/mnt/data/cyh/envs/vla-long-tail
SUITE=${SUITE:-libero_spatial}
DATA_MIX=${DATA_MIX:-libero_spatial_lt}
UNNORM_KEY=${UNNORM_KEY:-libero_spatial_lt}
CORRECT_STEPS=${CORRECT_STEPS:-100}
NUM_TRIALS=${NUM_TRIALS:-10}
ANCHOR_L2=${ANCHOR_L2:-10.0}
TCAD_LAMBDA=${TCAD_LAMBDA:-0.1}
TCAD_RATIO=${TCAD_RATIO:-0.5}
RARE_WEIGHT=${RARE_WEIGHT:-2.0}
RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}
RUN_ROOT=runs/spatial_lt_cgrbc
RESULT_ROOT=results/spatial_lt_cgrbc
LOG=/mnt/data/cyh/spatial_lt_cgrbc_screen_${RUN_STAMP}.log

wait_for_gpus() {
  echo "== waiting for GPUs 2/3 $(date -Is) =="
  while true; do
    local used2 used3 util2 util3
    used2=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 2 | tr -dc '0-9')
    used3=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3 | tr -dc '0-9')
    util2=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 2 | tr -dc '0-9')
    util3=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 3 | tr -dc '0-9')
    echo "$(date -Is) gpu2=${used2}MiB/${util2}% gpu3=${used3}MiB/${util3}%"
    if [[ "${used2:-999999}" -lt 2000 && "${used3:-999999}" -lt 2000 && "${util2:-999999}" -lt 20 && "${util3:-999999}" -lt 20 ]]; then
      break
    fi
    sleep 900
  done
}

link_eval_steps() {
  local run_id="$1"
  local step="$2"
  local ckpt_file ckpt_base suffix alias dest
  ckpt_file=$(find "${RUN_ROOT}/${run_id}/checkpoints" -maxdepth 1 -type f -name 'step-*.pt' | sort | tail -n 1)
  if [[ -z "${ckpt_file}" ]]; then
    echo "No real checkpoint found for ${run_id}" >&2
    return 1
  fi
  ckpt_base=$(basename "${ckpt_file}")
  suffix="${ckpt_base#step-}"
  suffix="${suffix#*-}"
  for alias in "${step}" "$(printf "%04d" "${step}")" "$(printf "%05d" "${step}")" "$(printf "%06d" "${step}")"; do
    dest="step-${alias}-${suffix}"
    if [[ "${dest}" != "${ckpt_base}" ]]; then
      ln -sf "${ckpt_base}" "${RUN_ROOT}/${run_id}/checkpoints/${dest}"
    fi
  done
}

run_eval() {
  local run_id="$1"
  local step="$2"
  local label="$3"
  local save_root="${RESULT_ROOT}/${run_id}/step${step}/${label}_${SUITE}_${NUM_TRIALS}trials_egl_$(date +%Y%m%d_%H%M%S)"
  echo "== eval start $(date -Is) label=${label} save_root=${save_root} =="
  python vla_scripts/parallel_libero_evaluator_egl.py \
    --num-trails-per-task "${NUM_TRIALS}" \
    --num-gpus 2 \
    --num-processes 10 \
    --task-suite-name "${SUITE}" \
    --pretrained-checkpoint "${RUN_ROOT}/${run_id}" \
    --unnorm_key "${UNNORM_KEY}" \
    --save-root "${save_root}" \
    --steps "${step}" \
    --instruction-formatting False
  echo "== eval done $(date -Is) label=${label} =="
  grep -R -E "Overall success rate|Task .*success rate" "${save_root}" 2>/dev/null || true
}

run_seed() {
  local seed="$1"
  local base_run="$2"
  local base_ckpt
  base_ckpt=$(find "${base_run}/checkpoints" -maxdepth 1 -type f -name 'step-*.pt' | sort | tail -n 1)
  if [[ -z "${base_ckpt}" ]]; then
    echo "Missing baseline checkpoint: ${base_run}" >&2
    return 2
  fi
  local run_id="cgrbc_${DATA_MIX}_base1000p${CORRECT_STEPS}_seed${seed}_a${ANCHOR_L2//./p}_${RUN_STAMP}"
  local smoke_id="cgrbc_smoke_${DATA_MIX}_base1000p5_seed${seed}_a${ANCHOR_L2//./p}_${RUN_STAMP}"
  echo "== seed ${seed} baseline=${base_run} base_ckpt=${base_ckpt} =="

  torchrun --nnodes 1 --nproc-per-node 2 --master_addr 127.0.0.1 --master_port $((30200 + seed)) vla_scripts/train.py \
    --pretrained_checkpoint "${base_ckpt}" \
    --vla.type "prism-qwen25-dinosiglip-224px+0_5b+mx-libero-core-lt" \
    --vla.data_mix "${DATA_MIX}" \
    --vla.expected_world_size 2 \
    --vla.global_batch_size 20 \
    --vla.per_device_batch_size 10 \
    --data_root_dir /mnt/data/cyh/tensorflow_datasets \
    --run_root_dir "${RUN_ROOT}" \
    --run_id "${smoke_id}" \
    --save_interval 5 \
    --seed "${seed}" \
    --tcad_lambda "${TCAD_LAMBDA}" \
    --tcad_ratio "${TCAD_RATIO}" \
    --tcad_margin 0.2 \
    --tcad_tail_max_count 9 \
    --tcad_conf_gate batch_median \
    --tcad_negative_mode manifest \
    --rare_bc_max_count 9 \
    --rare_bc_weight "${RARE_WEIGHT}" \
    --anchor_l2_lambda "${ANCHOR_L2}" \
    --anchor_l2_filter llm_backbone \
    --train_limit_steps 5
  echo "== smoke done seed=${seed} =="
  tail -n 20 "${RUN_ROOT}/${smoke_id}/tcad-debug.csv" 2>/dev/null || true

  torchrun --nnodes 1 --nproc-per-node 2 --master_addr 127.0.0.1 --master_port $((30300 + seed)) vla_scripts/train.py \
    --pretrained_checkpoint "${base_ckpt}" \
    --vla.type "prism-qwen25-dinosiglip-224px+0_5b+mx-libero-core-lt" \
    --vla.data_mix "${DATA_MIX}" \
    --vla.expected_world_size 2 \
    --vla.global_batch_size 20 \
    --vla.per_device_batch_size 10 \
    --data_root_dir /mnt/data/cyh/tensorflow_datasets \
    --run_root_dir "${RUN_ROOT}" \
    --run_id "${run_id}" \
    --save_interval "${CORRECT_STEPS}" \
    --seed "${seed}" \
    --tcad_lambda "${TCAD_LAMBDA}" \
    --tcad_ratio "${TCAD_RATIO}" \
    --tcad_margin 0.2 \
    --tcad_tail_max_count 9 \
    --tcad_conf_gate batch_median \
    --tcad_negative_mode manifest \
    --rare_bc_max_count 9 \
    --rare_bc_weight "${RARE_WEIGHT}" \
    --anchor_l2_lambda "${ANCHOR_L2}" \
    --anchor_l2_filter llm_backbone \
    --train_limit_steps "${CORRECT_STEPS}"
  echo "== correction done seed=${seed} run_id=${run_id} =="
  tail -n 20 "${RUN_ROOT}/${run_id}/tcad-debug.csv" 2>/dev/null || true
  link_eval_steps "${run_id}" "${CORRECT_STEPS}"
  run_eval "${run_id}" "${CORRECT_STEPS}" "cgrbc_seed${seed}"
}

{
  echo "== CGRBC screen start $(date -Is) stamp=${RUN_STAMP} =="
  source /opt/miniconda3/etc/profile.d/conda.sh
  conda activate "${ENV}"
  cd "${ROOT}"
  export CUDA_VISIBLE_DEVICES=2,3
  export HF_ENDPOINT=https://hf-mirror.com
  export HF_HOME=/mnt/data/cyh/.cache/huggingface
  export TRANSFORMERS_CACHE=/mnt/data/cyh/.cache/huggingface
  export TOKENIZERS_PARALLELISM=false
  export WANDB_DISABLED=true
  export PRISMATIC_DATA_ROOT="${ROOT}/data/prismatic"
  export PYTHONPATH="${ROOT}/LIBERO:${ROOT}:${PYTHONPATH:-}"
  export MUJOCO_GL=egl
  export PYTHONUNBUFFERED=1
  export HF_HUB_DISABLE_TELEMETRY=1
  export TFDS_DATA_DIR=/mnt/data/cyh/tensorflow_datasets
  export RARE_BC_CONFUSION_ONLY=1
  unset TARGET_TASK_INSTRUCTION
  unset TARGET_TASK_WEIGHT
  mkdir -p "${PRISMATIC_DATA_ROOT}" "${HF_HOME}" "${RUN_ROOT}" "${RESULT_ROOT}" autoresearch/state autoresearch/logs
  : > .hf_token

  wait_for_gpus
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
  run_seed 7 runs/spatial_lt_screen/baseline_libero_spatial_lt_s1000_seed7_b20
  run_seed 13 runs/spatial_lt_multiseed/baseline_libero_spatial_lt_s1000_seed13_b20_20260718_185539
  echo "== CGRBC screen all done $(date -Is) =="
} >> "${LOG}" 2>&1
