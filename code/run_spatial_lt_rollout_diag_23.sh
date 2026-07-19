#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/data/cyh/VLA-long-tail
ENV=/mnt/data/cyh/envs/vla-long-tail
RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}
LOG=/mnt/data/cyh/spatial_lt_rollout_diag_${RUN_STAMP}.log
SAVE_ROOT=results/spatial_lt_rollout_diag/baseline_seed7_${RUN_STAMP}
DIAG_ROOT=/mnt/data/cyh/VLA-long-tail/results/spatial_lt_rollout_diag/diag_seed7_${RUN_STAMP}
BASE_RUN=runs/spatial_lt_screen/baseline_libero_spatial_lt_s1000_seed7_b20

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

{
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
  export ROLLOUT_DIAG_FILE="${DIAG_ROOT}/rollout_diag.jsonl"
  export ROLLOUT_DIAG_DIR="${DIAG_ROOT}/frames"
  mkdir -p "${DIAG_ROOT}" "${SAVE_ROOT}"
  wait_for_gpus
  python /mnt/data/cyh/patch_rollout_diag_evaluator.py
  python - <<'PY'
from pathlib import Path
p = Path('vla_scripts/parallel_libero_evaluator_egl.py')
compile(p.read_text(encoding='utf-8'), str(p), 'exec')
print('compile-ok', p)
PY
  python vla_scripts/parallel_libero_evaluator_egl.py \
    --num-trails-per-task 2 \
    --num-gpus 2 \
    --num-processes 4 \
    --task-suite-name libero_spatial \
    --pretrained-checkpoint "${BASE_RUN}" \
    --unnorm_key libero_spatial_lt \
    --save-root "${SAVE_ROOT}" \
    --steps 1000 \
    --instruction-formatting False
  echo "== diag file == ${ROLLOUT_DIAG_FILE}"
  /mnt/data/cyh/envs/vla-long-tail/bin/python - <<'PY'
import json, os
from pathlib import Path
p = Path(os.environ['ROLLOUT_DIAG_FILE'])
rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
print('rows', len(rows), 'success', sum(r['success'] for r in rows))
for task_id in sorted({r['task_id'] for r in rows}):
    subset = [r for r in rows if r['task_id'] == task_id]
    succ = sum(r['success'] for r in subset)
    close = [r['close_step'] for r in subset if r['close_step'] is not None]
    mean_close = None if not close else sum(close)/len(close)
    mean_steps = sum(r['steps'] for r in subset)/len(subset)
    print(task_id, 'n', len(subset), 'succ', succ, 'mean_close', mean_close, 'mean_steps', round(mean_steps, 1))
PY
} >> "${LOG}" 2>&1
