#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/data/cyh/VLA-long-tail
ENV=/mnt/data/cyh/envs/vla-long-tail
SUITE=${SUITE:-libero_spatial}
DATA_MIX=${DATA_MIX:-libero_spatial_lt}
UNNORM_KEY=${UNNORM_KEY:-libero_spatial_lt}
CORRECT_STEPS=${CORRECT_STEPS:-50}
NUM_TRIALS=${NUM_TRIALS:-10}
EVAL_INIT_IDS=${EVAL_INIT_IDS:-5,6,7,8,9,10,11,12,13,14}
ANCHOR_L2=${ANCHOR_L2:-2.0}
BP_LAMBDA=${BP_LAMBDA:-0.2}
BP_TEMPERATURE=${BP_TEMPERATURE:-1.0}
TCAD_LAMBDA=${TCAD_LAMBDA:-0.05}
RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}
RUN_ROOT=runs/spatial_lt_bpc_rsdf
RESULT_ROOT=results/spatial_lt_bpc_rsdf
LOG=/mnt/data/cyh/spatial_lt_bpc_rsdf_screen_${RUN_STAMP}.log
BP_MANIFEST=/mnt/data/cyh/spatial_lt_bpc_rsdf_manifest_${RUN_STAMP}.json

BASE7=runs/spatial_lt_screen/baseline_libero_spatial_lt_s1000_seed7_b20
RSDF7=runs/spatial_lt_selective_soup/rsdf_barc100_visionllm_a0p5
BASE13=runs/spatial_lt_multiseed/baseline_libero_spatial_lt_s1000_seed13_b20_20260718_185539
RSDF13=runs/spatial_lt_rsdf_multiseed/rsdf_visionllm_barc100_seed13_a0p5_20260718_185539

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

latest_ckpt() {
  find "$1/checkpoints" -maxdepth 1 -type f -name 'step-*.pt' | sort | tail -n 1
}

link_eval_steps() {
  local run_root="$1" run_id="$2" step="$3" ckpt_file ckpt_base suffix alias dest
  ckpt_file=$(find "${run_root}/${run_id}/checkpoints" -maxdepth 1 -type f -name 'step-*.pt' | sort | tail -n 1)
  [[ -n "${ckpt_file}" ]] || { echo "No checkpoint for ${run_root}/${run_id}" >&2; return 1; }
  ckpt_base=$(basename "${ckpt_file}")
  suffix="${ckpt_base#step-}"; suffix="${suffix#*-}"
  for alias in "${step}" "$(printf "%03d" "${step}")" "$(printf "%04d" "${step}")" "$(printf "%05d" "${step}")" "$(printf "%06d" "${step}")"; do
    dest="step-${alias}-${suffix}"
    [[ "${dest}" == "${ckpt_base}" ]] || ln -sf "${ckpt_base}" "${run_root}/${run_id}/checkpoints/${dest}"
  done
}

make_bp_manifest() {
  python - "${BP_MANIFEST}" <<'PY'
import json, sys
from collections import defaultdict
from pathlib import Path

out = Path(sys.argv[1])
root = Path('results/spatial_lt_rollout_diag_pair')
pairs = [("baseline_seed7", "rsdf_seed7"), ("baseline_seed13", "rsdf_seed13")]
task_counts = {
    "pick up the black bowl between the plate and the ramekin and place it on the plate": 44,
    "pick up the black bowl from table center and place it on the plate": 28,
    "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate": 19,
    "pick up the black bowl next to the cookie box and place it on the plate": 15,
    "pick up the black bowl next to the plate and place it on the plate": 11,
    "pick up the black bowl next to the ramekin and place it on the plate": 9,
    "pick up the black bowl on the cookie box and place it on the plate": 8,
    "pick up the black bowl on the ramekin and place it on the plate": 7,
    "pick up the black bowl on the stove and place it on the plate": 6,
    "pick up the black bowl on the wooden cabinet and place it on the plate": 5,
}
by_task = defaultdict(list)

def load(label):
    path = root / label / 'diag_20260719_173523' / 'rollout_diag.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    stats = {}
    for task in sorted({row['task'] for row in rows}):
        sub = [row for row in rows if row['task'] == task]
        close = [row['close_step'] for row in sub if row['close_step'] is not None]
        stats[task] = {
            'steps': sum(row['steps'] for row in sub) / len(sub),
            'norm': sum(row['mean_action_norm'] for row in sub) / len(sub),
            'close': None if not close else sum(close) / len(close),
        }
    return stats

for base_label, rsdf_label in pairs:
    base, rsdf = load(base_label), load(rsdf_label)
    for task, b in base.items():
        r = rsdf.get(task)
        if r is None:
            continue
        steps_delta = r['steps'] - b['steps']
        norm_delta = r['norm'] - b['norm']
        close_delta = 0.0 if r['close'] is None or b['close'] is None else r['close'] - b['close']
        risk = 0.0
        risk += max(0.0, steps_delta) / 50.0
        risk += max(0.0, -norm_delta) / 0.10
        risk += max(0.0, abs(close_delta) - 8.0) / 30.0
        by_task[task].append({'risk': risk, 'steps_delta': steps_delta, 'norm_delta': norm_delta, 'close_delta': close_delta})

tasks = {}
for task in sorted(task_counts):
    vals = by_task.get(task, [])
    risk_score = 0.0 if not vals else sum(v['risk'] for v in vals) / len(vals)
    if risk_score >= 0.50:
        bp_weight = 1.0
    elif risk_score >= 0.20:
        bp_weight = 0.5
    elif risk_score >= 0.05:
        bp_weight = 0.25
    else:
        bp_weight = 0.0
    tail = task_counts[task] <= 9
    tasks[task] = {
        'task_count': task_counts[task],
        'risk_score': round(risk_score, 6),
        'bp_weight': bp_weight,
        'tcad_enable': bool(tail and bp_weight < 1.0),
        'bc_weight': 1.0,
        'deltas': vals,
    }
manifest = {
    'meta': {
        'method': 'BPC-RSDF',
        'source': 'paired_rollout_diag',
        'uses_success_label': False,
        'rule': 'risk gates baseline-preservation KL, not BC replay upweighting',
    },
    'tasks': tasks,
}
out.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print(json.dumps({'manifest': str(out), 'protected_tasks': {k:v['bp_weight'] for k,v in tasks.items() if v['bp_weight'] > 0}, 'tcad_tasks': [k for k,v in tasks.items() if v['tcad_enable']]}, indent=2))
PY
}

run_train() {
  local seed="$1" source_run="$2" teacher_run="$3" run_id="$4" steps="$5" port="$6"
  local source_ckpt teacher_ckpt
  source_ckpt=$(latest_ckpt "${source_run}")
  teacher_ckpt=$(latest_ckpt "${teacher_run}")
  [[ -n "${source_ckpt}" ]] || { echo "Missing source checkpoint for ${source_run}" >&2; return 2; }
  [[ -n "${teacher_ckpt}" ]] || { echo "Missing teacher checkpoint for ${teacher_run}" >&2; return 2; }
  echo "== BPC-RSDF train start $(date -Is) seed=${seed} source=${source_run} teacher=${teacher_run} run_id=${run_id} steps=${steps} =="
  torchrun --nnodes 1 --nproc-per-node 2 --master_addr 127.0.0.1 --master_port "${port}" vla_scripts/train.py \
    --pretrained_checkpoint "${source_ckpt}" \
    --baseline_teacher_checkpoint "${teacher_ckpt}" \
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
    --tcad_lambda "${TCAD_LAMBDA}" \
    --tcad_ratio 0.25 \
    --tcad_margin 0.2 \
    --tcad_tail_max_count 9 \
    --tcad_conf_gate batch_median \
    --tcad_negative_mode manifest \
    --tcad_detach_positive True \
    --rare_bc_weight 1.0 \
    --bp_preserve_manifest "${BP_MANIFEST}" \
    --bp_lambda "${BP_LAMBDA}" \
    --bp_temperature "${BP_TEMPERATURE}" \
    --bp_teacher_device cpu \
    --anchor_l2_lambda "${ANCHOR_L2}" \
    --anchor_l2_filter llm_backbone,vision_backbone \
    --train_limit_steps "${steps}"
  echo "== BPC-RSDF train done $(date -Is) run_id=${run_id} =="
  tail -n 20 "${RUN_ROOT}/${run_id}/tcad-debug.csv" 2>/dev/null || true
}

run_eval() {
  local label="$1" checkpoint="$2" step="$3"
  local save_root="${RESULT_ROOT}/${label}/step${step}/${label}_${SUITE}_${NUM_TRIALS}trials_initheldout_${RUN_STAMP}"
  echo "== eval start $(date -Is) label=${label} init_ids=${EVAL_INIT_IDS} save_root=${save_root} =="
  export CUDA_VISIBLE_DEVICES=2,3
  export EVAL_ALLOWED_GPUS=2,3
  export EVAL_INIT_IDS="${EVAL_INIT_IDS}"
  python vla_scripts/parallel_libero_evaluator_egl.py \
    --num-trails-per-task "${NUM_TRIALS}" \
    --num-gpus 2 \
    --num-processes 10 \
    --task-suite-name "${SUITE}" \
    --pretrained-checkpoint "${checkpoint}" \
    --unnorm_key "${UNNORM_KEY}" \
    --save-root "${save_root}" \
    --steps "${step}" \
    --instruction-formatting False
  grep -R -E "Overall success rate|Task .*success rate|Init ids" "${save_root}" 2>/dev/null || true
}

{
  echo "== BPC-RSDF screen start $(date -Is) stamp=${RUN_STAMP} =="
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
  export PRISMATIC_DATA_ROOT="${ROOT}/data/prismatic"
  export PYTHONPATH="${ROOT}/LIBERO:${ROOT}:${PYTHONPATH:-}"
  export MUJOCO_GL=egl
  export PYTHONUNBUFFERED=1
  export HF_HUB_DISABLE_TELEMETRY=1
  export TFDS_DATA_DIR=/mnt/data/cyh/tensorflow_datasets
  mkdir -p "${PRISMATIC_DATA_ROOT}" "${HF_HOME}" "${RUN_ROOT}" "${RESULT_ROOT}"
  : > .hf_token

  wait_for_gpus
  cp /mnt/data/cyh/bpc_datasets.py prismatic/vla/datasets/datasets.py
  cp /mnt/data/cyh/bpc_data_utils.py prismatic/util/data_utils.py
  cp /mnt/data/cyh/bpc_base_strategy.py prismatic/training/strategies/base_strategy.py
  cp /mnt/data/cyh/bpc_train.py vla_scripts/train.py
  python /mnt/data/cyh/patch_eval_fixed_init_ids.py
  python - <<'PY'
from pathlib import Path
for p in [Path('prismatic/vla/datasets/datasets.py'), Path('prismatic/util/data_utils.py'), Path('prismatic/training/strategies/base_strategy.py'), Path('vla_scripts/train.py'), Path('vla_scripts/parallel_libero_evaluator_egl.py')]:
    compile(p.read_text(encoding='utf-8'), str(p), 'exec')
    print('compile-ok', p)
PY
  make_bp_manifest

  smoke7="bpc_rsdf_smoke_${DATA_MIX}_seed7_p5_${RUN_STAMP}"
  smoke13="bpc_rsdf_smoke_${DATA_MIX}_seed13_p5_${RUN_STAMP}"
  run_train 7 "${RSDF7}" "${BASE7}" "${smoke7}" 5 31207
  run_train 13 "${RSDF13}" "${BASE13}" "${smoke13}" 5 31213

  bpc7="bpc_rsdf_${DATA_MIX}_seed7_p${CORRECT_STEPS}_${RUN_STAMP}"
  bpc13="bpc_rsdf_${DATA_MIX}_seed13_p${CORRECT_STEPS}_${RUN_STAMP}"
  run_train 7 "${RSDF7}" "${BASE7}" "${bpc7}" "${CORRECT_STEPS}" 31307
  run_train 13 "${RSDF13}" "${BASE13}" "${bpc13}" "${CORRECT_STEPS}" 31313
  link_eval_steps "${RUN_ROOT}" "${bpc7}" "${CORRECT_STEPS}"
  link_eval_steps "${RUN_ROOT}" "${bpc13}" "${CORRECT_STEPS}"

  run_eval baseline_seed7_heldout "${BASE7}" 1000
  run_eval rsdf_seed7_heldout "${RSDF7}" 100
  run_eval bpc_rsdf_seed7_heldout "${RUN_ROOT}/${bpc7}" "${CORRECT_STEPS}"
  run_eval baseline_seed13_heldout "${BASE13}" 1000
  run_eval rsdf_seed13_heldout "${RSDF13}" 100
  run_eval bpc_rsdf_seed13_heldout "${RUN_ROOT}/${bpc13}" "${CORRECT_STEPS}"
  echo "== BPC-RSDF screen all done $(date -Is) manifest=${BP_MANIFEST} =="
} >> "${LOG}" 2>&1