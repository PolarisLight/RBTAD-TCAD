#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/data/cyh/VLA-long-tail
ENV=/mnt/data/cyh/envs/vla-long-tail
RUN_ROOT=runs/core_full_protocol
RESULT_ROOT=results/core_full_protocol
LOG=/mnt/data/cyh/core_full_full_protocol_23.log
SCREEN_PID_FILE=/mnt/data/cyh/core_full_probe_eval30.pid
MIN_FULL_STEP=100000

wait_for_screening_eval() {
  if [[ ! -f "${SCREEN_PID_FILE}" ]]; then
    return 0
  fi

  local pid
  pid=$(cat "${SCREEN_PID_FILE}" 2>/dev/null || true)
  if [[ -z "${pid}" ]]; then
    return 0
  fi

  while kill -0 "${pid}" 2>/dev/null; do
    echo "== waiting for screening eval pid=${pid} before full protocol $(date -Is) =="
    sleep 600
  done
}

latest_step_for_run() {
  local run_id="$1"
  python - "$RUN_ROOT/$run_id/checkpoints" <<'PY'
import re
import sys
from pathlib import Path

ckpt_dir = Path(sys.argv[1])
best = None
for path in ckpt_dir.glob("step-*-epoch-*-loss=*.pt"):
    m = re.search(r"step-(\d+)-", path.name)
    if not m:
        continue
    step = int(m.group(1))
    if best is None or step > best:
        best = step
if best is None:
    raise SystemExit("no checkpoints found")
print(best)
PY
}

normalize_step_symlinks() {
  local run_id="$1"
  local step="$2"
  local ckpt_dir="${RUN_ROOT}/${run_id}/checkpoints"
  local ckpt_file
  ckpt_file=$(find "${ckpt_dir}" -maxdepth 1 -type f -name "step-$(printf "%06d" "${step}")-*.pt" | sort | tail -1)
  if [[ -z "${ckpt_file}" ]]; then
    ckpt_file=$(find "${ckpt_dir}" -maxdepth 1 -type f -name "step-*-${step}-*.pt" | sort | tail -1)
  fi
  if [[ -z "${ckpt_file}" ]]; then
    ckpt_file=$(find "${ckpt_dir}" -maxdepth 1 -type f -name "step-*.pt" | sort | tail -1)
  fi
  local ckpt_base
  ckpt_base=$(basename "${ckpt_file}")
  local suffix
  suffix=$(echo "${ckpt_base}" | sed -E 's/^step-[0-9]+-//')
  ln -sf "${ckpt_base}" "${ckpt_dir}/step-${step}-${suffix}"
  ln -sf "${ckpt_base}" "${ckpt_dir}/step-$(printf "%05d" "${step}")-${suffix}"
  ln -sf "${ckpt_base}" "${ckpt_dir}/step-$(printf "%06d" "${step}")-${suffix}"
}

train_one() {
  local mode="$1"
  local port="$2"
  local run_id
  local tcad_lambda="0.0"
  local tcad_ratio="0.0"
  local tcad_tail_max_count="0"
  local tcad_conf_gate="none"

  if [[ "${mode}" == "baseline" ]]; then
    run_id=baseline_libero_core_full_protocol_seed7_b20
  else
    run_id=rbtad_libero_core_full_alltcad_protocol_seed7_b20
    tcad_lambda="0.1"
    tcad_ratio="0.5"
    tcad_tail_max_count="1000000000"
    tcad_conf_gate="batch_median"
  fi

  echo "== ${mode} full train start $(date -Is) run_id=${run_id} =="
  echo "== command: core_full 36 epochs, seed=7, global_batch=20, gpus=2 =="
  set +e
  torchrun --nnodes 1 --nproc-per-node 2 --master_addr 127.0.0.1 --master_port "${port}" vla_scripts/train.py \
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
    --tcad_lambda "${tcad_lambda}" \
    --tcad_ratio "${tcad_ratio}" \
    --tcad_margin 0.2 \
    --tcad_tail_max_count "${tcad_tail_max_count}" \
    --tcad_conf_gate "${tcad_conf_gate}" \
    --rare_bc_max_count 0 \
    --rare_bc_weight 1.0
  local train_status=$?
  set -e

  echo "== ${mode} full train exit $(date -Is) status=${train_status} =="
  find "${RUN_ROOT}/${run_id}/checkpoints" -maxdepth 1 -type f -name 'step-*.pt' -printf "%f %s\n" | sort || true
  tail -30 "${RUN_ROOT}/${run_id}/metrics.jsonl" 2>/dev/null || true
  tail -30 "${RUN_ROOT}/${run_id}/tcad-debug.csv" 2>/dev/null || true

  if [[ "${train_status}" != "0" ]]; then
    if find "${RUN_ROOT}/${run_id}/checkpoints" -maxdepth 1 -type f -name 'step-*.pt' | grep -q .; then
      echo "torchrun exited with status ${train_status} after saving checkpoint; continuing to final eval"
    else
      echo "torchrun failed with status ${train_status} and no checkpoint exists"
      exit "${train_status}"
    fi
  fi

  local step
  step=$(latest_step_for_run "${run_id}")
  if (( step < MIN_FULL_STEP )); then
    echo "latest checkpoint step ${step} is below full-protocol threshold ${MIN_FULL_STEP}; aborting eval"
    exit 1
  fi
  normalize_step_symlinks "${run_id}" "${step}"
  echo "${step}" > "${RUN_ROOT}/${run_id}/selected_eval_step.txt"
}

eval_one() {
  local mode="$1"
  local run_id
  if [[ "${mode}" == "baseline" ]]; then
    run_id=baseline_libero_core_full_protocol_seed7_b20
  else
    run_id=rbtad_libero_core_full_alltcad_protocol_seed7_b20
  fi

  local step
  step=$(cat "${RUN_ROOT}/${run_id}/selected_eval_step.txt")
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

{
  echo "== core_full full protocol queue start $(date -Is) =="
  wait_for_screening_eval

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
  train_one baseline 29621
  eval_one baseline
  train_one rbtad 29622
  eval_one rbtad
  echo "== core_full full protocol all done $(date -Is) =="
} >> "${LOG}" 2>&1
