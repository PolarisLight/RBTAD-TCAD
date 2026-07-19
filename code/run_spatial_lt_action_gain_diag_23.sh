#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/data/cyh/VLA-long-tail
ENV=/mnt/data/cyh/envs/vla-long-tail
SUITE=${SUITE:-libero_spatial}
UNNORM_KEY=${UNNORM_KEY:-libero_spatial_lt}
NUM_TRIALS=${NUM_TRIALS:-10}
GAIN=${GAIN:-1.15}
TAIL_TASK_IDS=${TAIL_TASK_IDS:-1,3,5,7,9}
RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}
RESULT_ROOT=results/spatial_lt_action_gain_diag
LOG=/mnt/data/cyh/spatial_lt_action_gain_diag_${RUN_STAMP}.log

wait_for_gpus() {
  echo "== waiting for GPUs 2/3 $(date -Is) =="
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

run_eval() {
  local seed="$1" checkpoint="$2" label="$3" gain="$4"
  local save_root="${RESULT_ROOT}/${label}_seed${seed}_gain${gain//./p}_${NUM_TRIALS}trials_${RUN_STAMP}"
  echo "== eval start $(date -Is) seed=${seed} label=${label} gain=${gain} save_root=${save_root} =="
  export TAIL_ACTION_GAIN="${gain}"
  export TAIL_ACTION_TASK_IDS="${TAIL_TASK_IDS}"
  export EVAL_ALLOWED_GPUS=2,3
  python vla_scripts/parallel_libero_evaluator_egl.py \
    --num-trails-per-task "${NUM_TRIALS}" \
    --num-gpus 2 \
    --num-processes 10 \
    --task-suite-name "${SUITE}" \
    --pretrained-checkpoint "${checkpoint}" \
    --unnorm_key "${UNNORM_KEY}" \
    --save-root "${save_root}" \
    --steps 1000 \
    --instruction-formatting False
  grep -R -E "Overall success rate|Task .*success rate" "${save_root}" 2>/dev/null || true
}

{
  echo "== Tail action-gain diagnostic start $(date -Is) stamp=${RUN_STAMP} gain=${GAIN} tail_ids=${TAIL_TASK_IDS} =="
  source /opt/miniconda3/etc/profile.d/conda.sh
  conda activate "${ENV}"
  cd "${ROOT}"
  export CUDA_VISIBLE_DEVICES=2,3
  export EVAL_ALLOWED_GPUS=2,3
  export HF_ENDPOINT=https://hf-mirror.com
  export HF_HOME=/mnt/data/cyh/.cache/huggingface
  export TRANSFORMERS_CACHE=/mnt/data/cyh/.cache/huggingface
  export TOKENIZERS_PARALLELISM=false
  export WANDB_DISABLED=true
  export MUJOCO_GL=egl
  export PYTHONPATH="${ROOT}/LIBERO:${ROOT}:${PYTHONPATH:-}"
  export PYTHONUNBUFFERED=1
  export HF_HUB_DISABLE_TELEMETRY=1
  : > .hf_token
  mkdir -p "${RESULT_ROOT}"
  wait_for_gpus
  python /mnt/data/cyh/patch_tail_action_gain_evaluator.py
  python - <<'PY'
from pathlib import Path
p = Path('vla_scripts/parallel_libero_evaluator_egl.py')
compile(p.read_text(encoding='utf-8'), str(p), 'exec')
print('compile-ok', p)
PY
  run_eval 7 runs/spatial_lt_screen/baseline_libero_spatial_lt_s1000_seed7_b20 baseline 1.0
  run_eval 7 runs/spatial_lt_screen/baseline_libero_spatial_lt_s1000_seed7_b20 action_gain "${GAIN}"
  run_eval 13 runs/spatial_lt_multiseed/baseline_libero_spatial_lt_s1000_seed13_b20_20260718_185539 baseline 1.0
  run_eval 13 runs/spatial_lt_multiseed/baseline_libero_spatial_lt_s1000_seed13_b20_20260718_185539 action_gain "${GAIN}"
  echo "== Tail action-gain diagnostic all done $(date -Is) =="
} >> "${LOG}" 2>&1