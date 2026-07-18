#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/data/cyh/VLA-long-tail
ENV=/mnt/data/cyh/envs/vla-long-tail
SUITE=libero_spatial
DATA_MIX=libero_spatial_lt
UNNORM_KEY=libero_spatial_lt
CORRECT_STEPS=100
NUM_TRIALS=${NUM_TRIALS:-10}
ANCHOR_L2=10.0
RUN_STAMP=20260719_052007
RUN_ROOT=runs/spatial_lt_rlct
RESULT_ROOT=results/spatial_lt_rlct
LOG=/mnt/data/cyh/spatial_lt_rlct_recover_${RUN_STAMP}_$(date +%Y%m%d_%H%M%S).log

wait_for_gpus() {
  while true; do
    used2=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 2 | tr -dc '0-9')
    used3=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3 | tr -dc '0-9')
    util2=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 2 | tr -dc '0-9')
    util3=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 3 | tr -dc '0-9')
    echo "$(date -Is) gpu2=${used2}MiB/${util2}% gpu3=${used3}MiB/${util3}%"
    if [[ "${used2:-999999}" -lt 2000 && "${used3:-999999}" -lt 2000 && "${util2:-999999}" -lt 20 && "${util3:-999999}" -lt 20 ]]; then break; fi
    sleep 900
  done
}

latest_ckpt() {
  find "$1/checkpoints" -maxdepth 1 -type f -name 'step-*.pt' | sort | tail -n 1
}

link_eval_steps() {
  local run_id="$1"; local step="$2"; local ckpt_file ckpt_base suffix alias dest
  ckpt_file=$(latest_ckpt "${RUN_ROOT}/${run_id}")
  [[ -n "${ckpt_file}" ]] || { echo "No real checkpoint found for ${run_id}" >&2; return 1; }
  ckpt_base=$(basename "${ckpt_file}")
  suffix="${ckpt_base#step-}"; suffix="${suffix#*-}"
  for alias in "${step}" "$(printf "%03d" "${step}")" "$(printf "%04d" "${step}")" "$(printf "%05d" "${step}")" "$(printf "%06d" "${step}")"; do
    dest="step-${alias}-${suffix}"
    [[ "${dest}" == "${ckpt_base}" ]] || ln -sf "${ckpt_base}" "${RUN_ROOT}/${run_id}/checkpoints/${dest}"
  done
}

run_eval() {
  local run_id="$1"; local step="$2"; local label="$3"
  local save_root="${RESULT_ROOT}/${run_id}/step${step}/${label}_${SUITE}_${NUM_TRIALS}trials_egl_$(date +%Y%m%d_%H%M%S)"
  echo "== eval start $(date -Is) label=${label} save_root=${save_root} =="
  link_eval_steps "${run_id}" "${step}"
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

run_train_allow_saved() {
  local seed="$1"; local base_run="$2"; local run_id="$3"; local steps="$4"; local port="$5"
  local base_ckpt
  base_ckpt=$(latest_ckpt "${base_run}")
  [[ -n "${base_ckpt}" ]] || { echo "Missing baseline checkpoint ${base_run}" >&2; return 2; }
  echo "== train start $(date -Is) seed=${seed} run_id=${run_id} steps=${steps} =="
  set +e
  torchrun --nnodes 1 --nproc-per-node 2 --master_addr 127.0.0.1 --master_port "${port}" vla_scripts/train.py \
    --pretrained_checkpoint "${base_ckpt}" \
    --vla.type "prism-qwen25-dinosiglip-224px+0_5b+mx-libero-core-lt" \
    --vla.data_mix "${DATA_MIX}" \
    --vla.expected_world_size 2 \
    --vla.global_batch_size 20 \
    --vla.per_device_batch_size 10 \
    --data_root_dir /mnt/data/cyh/tensorflow_datasets \
    --run_root_dir "${RUN_ROOT}" \
    --run_id "${run_id}" \
    --save_interval "${steps}" \
    --seed "${seed}" \
    --tcad_lambda 0.1 \
    --tcad_ratio 0.5 \
    --tcad_margin 0.2 \
    --tcad_tail_max_count 9 \
    --tcad_conf_gate batch_median \
    --tcad_negative_mode manifest \
    --rare_bc_max_count 9 \
    --rare_bc_weight 2.0 \
    --anchor_l2_lambda "${ANCHOR_L2}" \
    --anchor_l2_filter llm_backbone \
    --trainable_filter llm_backbone \
    --train_limit_steps "${steps}"
  status=$?
  set -e
  ckpt=$(latest_ckpt "${RUN_ROOT}/${run_id}" || true)
  if [[ "${status}" -ne 0 ]]; then
    if [[ -n "${ckpt}" ]]; then
      echo "== train returned ${status} after checkpoint save; continuing with ${ckpt} =="
    else
      echo "== train failed before checkpoint, status=${status} ==" >&2
      return "${status}"
    fi
  fi
  tail -n 20 "${RUN_ROOT}/${run_id}/tcad-debug.csv" 2>/dev/null || true
}

{
  echo "== RLCT recover start $(date -Is) =="
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
  mkdir -p "${PRISMATIC_DATA_ROOT}" "${HF_HOME}" "${RUN_ROOT}" "${RESULT_ROOT}"
  : > .hf_token
  wait_for_gpus
  run_eval rlct_libero_spatial_lt_base1000p100_seed7_a10p0_20260719_052007 100 rlct_seed7_recover
  run_train_allow_saved 13 runs/spatial_lt_multiseed/baseline_libero_spatial_lt_s1000_seed13_b20_20260718_185539 rlct_smoke_libero_spatial_lt_base1000p5_seed13_a10p0_${RUN_STAMP} 5 30413
  run_train_allow_saved 13 runs/spatial_lt_multiseed/baseline_libero_spatial_lt_s1000_seed13_b20_20260718_185539 rlct_libero_spatial_lt_base1000p100_seed13_a10p0_${RUN_STAMP} 100 30513
  run_eval rlct_libero_spatial_lt_base1000p100_seed13_a10p0_${RUN_STAMP} 100 rlct_seed13_recover
  echo "== RLCT recover all done $(date -Is) =="
} >> "${LOG}" 2>&1
