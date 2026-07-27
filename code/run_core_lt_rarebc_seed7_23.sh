#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/data/cyh/VLA-long-tail
ENV=/mnt/data/cyh/envs/vla-long-tail
DATA_ROOT=/mnt/data/cyh/tensorflow_datasets
RUN_ROOT=runs/core_lt_ablation
RESULT_ROOT=results/core_lt_ablation
RUN_ID=rarebc_libero_core_lt_w3_tail9_seed7_b20
LOG=/mnt/data/cyh/core_lt_rarebc_seed7_20260728.log

wait_for_gpus() {
  echo "== wait for GPUs 2/3 $(date -Is) =="
  while true; do
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

latest_step_for_run() {
  python - "${RUN_ROOT}/${RUN_ID}/checkpoints" <<'PY'
import re, sys
from pathlib import Path
best = None
for p in Path(sys.argv[1]).glob("step-*.pt"):
    m = re.search(r"step-(\d+)-", p.name)
    if m:
        best = max(best or 0, int(m.group(1)))
if not best:
    raise SystemExit("no checkpoints found")
print(best)
PY
}

link_eval_steps() {
  local step="$1"
  local ckpt_dir="${RUN_ROOT}/${RUN_ID}/checkpoints"
  local ckpt_file ckpt_base suffix alias dest
  ckpt_file=$(find "${ckpt_dir}" -maxdepth 1 -type f -name "step-*.pt" | sort | tail -n 1)
  ckpt_base=$(basename "${ckpt_file}")
  suffix="${ckpt_base#step-}"
  suffix="${suffix#*-}"
  for alias in "${step}" "$(printf "%05d" "${step}")" "$(printf "%06d" "${step}")"; do
    dest="step-${alias}-${suffix}"
    [[ "${dest}" == "${ckpt_base}" ]] || ln -sf "${ckpt_base}" "${ckpt_dir}/${dest}"
  done
}

{
  echo "== Core-LT Rare-BC seed7 start $(date -Is) =="
  source /opt/miniconda3/etc/profile.d/conda.sh
  conda activate "${ENV}"
  cd "${ROOT}"
  export CUDA_VISIBLE_DEVICES=2,3
  export EVAL_ALLOWED_GPUS=2,3
  export MUJOCO_GL=egl
  export HF_ENDPOINT=https://hf-mirror.com
  export HF_HOME=/mnt/data/cyh/.cache/huggingface
  export TRANSFORMERS_CACHE=/mnt/data/cyh/.cache/huggingface
  export TOKENIZERS_PARALLELISM=false
  export WANDB_DISABLED=true
  export PRISMATIC_DATA_ROOT="${ROOT}/data/prismatic"
  export PYTHONPATH="${ROOT}/LIBERO:${ROOT}:${PYTHONPATH:-}"
  export PYTHONUNBUFFERED=1
  export TFDS_DATA_DIR="${DATA_ROOT}"
  mkdir -p "${PRISMATIC_DATA_ROOT}" "${HF_HOME}" "${RUN_ROOT}" "${RESULT_ROOT}"
  : > .hf_token

  wait_for_gpus
  if [[ ! -d "${RUN_ROOT}/${RUN_ID}/checkpoints" ]] || ! find "${RUN_ROOT}/${RUN_ID}/checkpoints" -maxdepth 1 -name "step-*.pt" | grep -q .; then
    echo "== train Rare-BC $(date -Is) =="
    torchrun --nnodes 1 --nproc-per-node 2 --master_addr 127.0.0.1 --master_port 31907 vla_scripts/train.py \
      --pretrained_checkpoint pretrained/minivla-libero90-prismatic/checkpoints/step-122500-epoch-55-loss=0.0743.pt \
      --vla.type "prism-qwen25-dinosiglip-224px+0_5b+mx-libero-core-lt" \
      --vla.data_mix libero_core_lt \
      --vla.expected_world_size 2 \
      --vla.global_batch_size 20 \
      --vla.per_device_batch_size 10 \
      --data_root_dir "${DATA_ROOT}" \
      --run_root_dir "${RUN_ROOT}" \
      --run_id "${RUN_ID}" \
      --save_interval 5000 \
      --seed 7 \
      --tcad_lambda 0.0 \
      --tcad_ratio 0.0 \
      --rare_bc_max_count 9 \
      --rare_bc_weight 3.0
  else
    echo "== skip train Rare-BC; checkpoint exists =="
  fi

  step=$(latest_step_for_run)
  link_eval_steps "${step}"
  wait_for_gpus
  save_root="${RESULT_ROOT}/${RUN_ID}/${step}/rarebc_seed7_libero_core_30trials_20260728"
  echo "== eval Rare-BC $(date -Is) step=${step} save_root=${save_root} =="
  python vla_scripts/parallel_libero_evaluator_egl.py \
    --num-trails-per-task 30 \
    --num-gpus 2 \
    --num-processes 10 \
    --task-suite-name libero_core \
    --pretrained-checkpoint "${RUN_ROOT}/${RUN_ID}" \
    --unnorm_key libero_core_lt \
    --save-root "${save_root}" \
    --steps "${step}" \
    --instruction-formatting False
  grep -R -E "Overall success rate|Task .*success rate" "${save_root}" 2>/dev/null || true
  echo "== Core-LT Rare-BC seed7 done $(date -Is) =="
} >> "${LOG}" 2>&1

