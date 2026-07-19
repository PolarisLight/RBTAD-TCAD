#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/data/cyh/VLA-long-tail
ENV=/mnt/data/cyh/envs/vla-long-tail
SUITE=${SUITE:-libero_spatial}
UNNORM_KEY=${UNNORM_KEY:-libero_spatial_lt}
NUM_TRIALS=${NUM_TRIALS:-5}
NUM_PROCS=${NUM_PROCS:-5}
EVAL_INIT_IDS=${EVAL_INIT_IDS:-0,1,2,3,4}
RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}
RESULT_ROOT=results/spatial_lt_rollout_diag_pair
LOG=/mnt/data/cyh/spatial_lt_rollout_diag_pair_${RUN_STAMP}.log

wait_for_gpus() {
  echo "== waiting for GPUs 2/3 $(date -Is) =="
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

run_diag() {
  local label="$1" checkpoint="$2" step="$3"
  local save_root="${RESULT_ROOT}/${label}/step${step}/${label}_${SUITE}_${NUM_TRIALS}trials_initfixed_${RUN_STAMP}"
  local diag_root="${ROOT}/${RESULT_ROOT}/${label}/diag_${RUN_STAMP}"
  echo "== rollout diag start $(date -Is) label=${label} ckpt=${checkpoint} step=${step} init_ids=${EVAL_INIT_IDS} =="
  export CUDA_VISIBLE_DEVICES=2,3
  export EVAL_ALLOWED_GPUS=2,3
  export EVAL_INIT_IDS="${EVAL_INIT_IDS}"
  export ROLLOUT_DIAG_FILE="${diag_root}/rollout_diag.jsonl"
  export ROLLOUT_DIAG_DIR="${diag_root}/frames"
  mkdir -p "${save_root}" "${diag_root}"
  python vla_scripts/parallel_libero_evaluator_egl.py \
    --num-trails-per-task "${NUM_TRIALS}" \
    --num-gpus 2 \
    --num-processes "${NUM_PROCS}" \
    --task-suite-name "${SUITE}" \
    --pretrained-checkpoint "${checkpoint}" \
    --unnorm_key "${UNNORM_KEY}" \
    --save-root "${save_root}" \
    --steps "${step}" \
    --instruction-formatting False
  echo "== rollout diag done $(date -Is) label=${label} diag=${ROLLOUT_DIAG_FILE} =="
  grep -R -E "Overall success rate|Task .*success rate|Init ids" "${save_root}" 2>/dev/null || true
}

summarize_diag() {
  python - <<'PY'
import json, os
from collections import defaultdict
from pathlib import Path

root = Path("results/spatial_lt_rollout_diag_pair")
files = sorted(root.glob("*/diag_*/rollout_diag.jsonl"))
print("diag_files", len(files))
for path in files:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    label = path.parts[-3]
    print(f"label={label} rows={len(rows)} success={sum(r['success'] for r in rows)} rate={sum(r['success'] for r in rows)/max(len(rows),1):.3f}")
    by_task = defaultdict(list)
    for row in rows:
        by_task[row["task_id"]].append(row)
    for task_id in sorted(by_task):
        subset = by_task[task_id]
        close = [r["close_step"] for r in subset if r["close_step"] is not None]
        mean_close = None if not close else round(sum(close) / len(close), 2)
        mean_steps = round(sum(r["steps"] for r in subset) / len(subset), 2)
        mean_norm = round(sum(r["mean_action_norm"] for r in subset) / len(subset), 4)
        succ = sum(r["success"] for r in subset)
        print(f"  task={task_id:02d} n={len(subset)} succ={succ} mean_close={mean_close} mean_steps={mean_steps} mean_action_norm={mean_norm}")
PY
}

{
  echo "== Spatial-LT rollout diagnostic pair start $(date -Is) stamp=${RUN_STAMP} =="
  source /opt/miniconda3/etc/profile.d/conda.sh
  conda activate "${ENV}"
  cd "${ROOT}"
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
  wait_for_gpus
  python /mnt/data/cyh/patch_eval_fixed_init_ids.py
  python /mnt/data/cyh/patch_rollout_diag_evaluator.py
  python - <<'PY'
from pathlib import Path
p = Path('vla_scripts/parallel_libero_evaluator_egl.py')
compile(p.read_text(encoding='utf-8'), str(p), 'exec')
print('compile-ok', p)
PY
  run_diag baseline_seed7 runs/spatial_lt_screen/baseline_libero_spatial_lt_s1000_seed7_b20 1000
  run_diag rsdf_seed7 runs/spatial_lt_selective_soup/rsdf_barc100_visionllm_a0p5 100
  run_diag baseline_seed13 runs/spatial_lt_multiseed/baseline_libero_spatial_lt_s1000_seed13_b20_20260718_185539 1000
  run_diag rsdf_seed13 runs/spatial_lt_rsdf_multiseed/rsdf_visionllm_barc100_seed13_a0p5_20260718_185539 100
  summarize_diag
  echo "== Spatial-LT rollout diagnostic pair all done $(date -Is) =="
} >> "${LOG}" 2>&1

