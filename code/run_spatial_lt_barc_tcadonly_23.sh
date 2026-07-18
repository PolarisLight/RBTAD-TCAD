#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/data/cyh/VLA-long-tail
ENV=/mnt/data/cyh/envs/vla-long-tail
SEED=${SEED:-7}
SUITE=${SUITE:-libero_spatial}
DATA_MIX=${DATA_MIX:-libero_spatial_lt}
UNNORM_KEY=${UNNORM_KEY:-libero_spatial_lt}
NUM_TRIALS=${NUM_TRIALS:-10}
CORRECT_STEPS=${CORRECT_STEPS:-50}
ANCHOR_L2=${ANCHOR_L2:-10.0}
TCAD_LAMBDA=${TCAD_LAMBDA:-0.1}
TCAD_RATIO=${TCAD_RATIO:-0.5}
PROBE_NAME=${PROBE_NAME:-spatial_lt_barc_tcadonly}
BASE_RUN=${BASE_RUN:-runs/spatial_lt_screen/baseline_libero_spatial_lt_s1000_seed7_b20}
BASE_STEP=${BASE_STEP:-1000}
RUN_ROOT=runs/spatial_lt_barc_tcadonly
RESULT_ROOT=results/spatial_lt_barc_tcadonly
RUN_ID="barc_tcadonly_${DATA_MIX}_base${BASE_STEP}p${CORRECT_STEPS}_seed${SEED}_b20_a${ANCHOR_L2//./p}"
SMOKE_ID="barc_tcadonly_smoke_${DATA_MIX}_base${BASE_STEP}p5_seed${SEED}_b20_a${ANCHOR_L2//./p}"
LOG=/mnt/data/cyh/${PROBE_NAME}_base${BASE_STEP}p${CORRECT_STEPS}_a${ANCHOR_L2//./p}_t${NUM_TRIALS}.log

link_eval_steps() {
  local run_id="$1"
  local step="$2"
  local ckpt_file ckpt_base suffix alias dest
  ckpt_file=$(find "${RUN_ROOT}/${run_id}/checkpoints" -maxdepth 1 -type f -name 'step-*.pt' | sort | tail -n 1)
  if [[ -z "${ckpt_file}" ]]; then
    echo "No real checkpoint found for ${run_id}"
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

safe_run_id() {
  local base="$1"
  if [[ -e "${RUN_ROOT}/${base}" ]]; then
    echo "${base}_rerun$(date +%Y%m%d_%H%M%S)"
  else
    echo "${base}"
  fi
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
  mkdir -p "$PRISMATIC_DATA_ROOT" "$HF_HOME" "$RUN_ROOT" "$RESULT_ROOT" autoresearch/state autoresearch/logs
  : > .hf_token

  echo "== gpu precheck $(date -Is) =="
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
  used2=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 2 | tr -dc '0-9')
  used3=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3 | tr -dc '0-9')
  if [[ "${used2:-999999}" -gt 2000 || "${used3:-999999}" -gt 2000 ]]; then
    echo "GPU 2/3 not free enough: gpu2=${used2}MiB gpu3=${used3}MiB"
    exit 3
  fi

  base_ckpt=$(find "${BASE_RUN}/checkpoints" -maxdepth 1 -type f -name 'step-*.pt' | sort | tail -n 1)
  echo "base_ckpt=${base_ckpt}"
  if [[ -z "${base_ckpt}" ]]; then
    echo "Missing baseline checkpoint under ${BASE_RUN}/checkpoints"
    exit 4
  fi

  SMOKE_ID=$(safe_run_id "${SMOKE_ID}")
  RUN_ID=$(safe_run_id "${RUN_ID}")
  echo "smoke_id=${SMOKE_ID}"
  echo "run_id=${RUN_ID}"

  echo "== TCAD-only 5-step smoke start $(date -Is) =="
  torchrun --nnodes 1 --nproc-per-node 2 --master_addr 127.0.0.1 --master_port 29853 vla_scripts/train.py \
    --pretrained_checkpoint "${base_ckpt}" \
    --vla.type "prism-qwen25-dinosiglip-224px+0_5b+mx-libero-core-lt" \
    --vla.data_mix "${DATA_MIX}" \
    --vla.expected_world_size 2 \
    --vla.global_batch_size 20 \
    --vla.per_device_batch_size 10 \
    --data_root_dir /mnt/data/cyh/tensorflow_datasets \
    --run_root_dir "${RUN_ROOT}" \
    --run_id "${SMOKE_ID}" \
    --save_interval 5 \
    --seed "${SEED}" \
    --tcad_lambda "${TCAD_LAMBDA}" \
    --tcad_ratio "${TCAD_RATIO}" \
    --tcad_margin 0.2 \
    --tcad_tail_max_count 9 \
    --tcad_conf_gate batch_median \
    --tcad_negative_mode manifest \
    --rare_bc_max_count 0 \
    --rare_bc_weight 1.0 \
    --anchor_l2_lambda "${ANCHOR_L2}" \
    --anchor_l2_filter llm_backbone \
    --train_limit_steps 5
  echo "== TCAD-only smoke done $(date -Is) =="
  tail -n 20 "${RUN_ROOT}/${SMOKE_ID}/tcad-debug.csv" 2>/dev/null || true

  echo "== TCAD-only ${CORRECT_STEPS}-step correction start $(date -Is) =="
  torchrun --nnodes 1 --nproc-per-node 2 --master_addr 127.0.0.1 --master_port 29854 vla_scripts/train.py \
    --pretrained_checkpoint "${base_ckpt}" \
    --vla.type "prism-qwen25-dinosiglip-224px+0_5b+mx-libero-core-lt" \
    --vla.data_mix "${DATA_MIX}" \
    --vla.expected_world_size 2 \
    --vla.global_batch_size 20 \
    --vla.per_device_batch_size 10 \
    --data_root_dir /mnt/data/cyh/tensorflow_datasets \
    --run_root_dir "${RUN_ROOT}" \
    --run_id "${RUN_ID}" \
    --save_interval "${CORRECT_STEPS}" \
    --seed "${SEED}" \
    --tcad_lambda "${TCAD_LAMBDA}" \
    --tcad_ratio "${TCAD_RATIO}" \
    --tcad_margin 0.2 \
    --tcad_tail_max_count 9 \
    --tcad_conf_gate batch_median \
    --tcad_negative_mode manifest \
    --rare_bc_max_count 0 \
    --rare_bc_weight 1.0 \
    --anchor_l2_lambda "${ANCHOR_L2}" \
    --anchor_l2_filter llm_backbone \
    --train_limit_steps "${CORRECT_STEPS}"
  echo "== TCAD-only correction done $(date -Is) =="
  tail -n 20 "${RUN_ROOT}/${RUN_ID}/tcad-debug.csv" 2>/dev/null || true

  link_eval_steps "${RUN_ID}" "${CORRECT_STEPS}"
  stamp=$(date +%Y%m%d_%H%M%S)
  save_root="${RESULT_ROOT}/${RUN_ID}/base${BASE_STEP}p${CORRECT_STEPS}/tcadonly_${SUITE}_${NUM_TRIALS}trials_egl_${stamp}"
  echo "== TCAD-only eval start $(date -Is) =="
  echo "save_root=${save_root}"
  python vla_scripts/parallel_libero_evaluator_egl.py \
    --num-trails-per-task "${NUM_TRIALS}" \
    --num-gpus 2 \
    --num-processes 10 \
    --task-suite-name "${SUITE}" \
    --pretrained-checkpoint "${RUN_ROOT}/${RUN_ID}" \
    --unnorm_key "${UNNORM_KEY}" \
    --save-root "${save_root}" \
    --steps "${CORRECT_STEPS}" \
    --instruction-formatting False
  echo "== TCAD-only eval done $(date -Is) =="
  grep -R -E "Overall success rate|Task .*success rate" "${save_root}" 2>/dev/null || true
  echo "== ${PROBE_NAME} all done $(date -Is) =="
} >> "$LOG" 2>&1