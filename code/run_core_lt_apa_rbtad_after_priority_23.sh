#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/data/cyh/VLA-long-tail
ENV=/mnt/data/cyh/envs/vla-long-tail
LOG=/mnt/data/cyh/core_lt_apa_rbtad_after_priority_20260728.log
GPU_IDS=2,3
PRETRAIN=pretrained/minivla-libero90-prismatic/checkpoints/step-122500-epoch-55-loss=0.0743.pt
VLA_TYPE=prism-qwen25-dinosiglip-224px+0_5b+mx-libero-core-lt
DATA_MIX=libero_core_apa
TASK_SUITE=libero_core
UNNORM_KEY=libero_core_apa
RUN_GROUP=core_lt_complementarity
RUN_ID=apa_rbtad_libero_core_apa_w3_tail9_confmedian_seed7_b20

log() { echo "[$(date -Is)] $*"; }

wait_for_priority_front() {
  log "waiting for Rare-BC, same-pipeline APA, and multiseed priority queue"
  while pgrep -af 'run_core_lt_rarebc_seed7_23.sh|rarebc_libero_core_lt_w3_tail9_seed7_b20|run_core_all_hf_download_then_apa_seed7_23.sh|core_all_hf_apa_seed7|run_core_lt_priority_after_apa_multiseed_23.sh' >/dev/null; do
    pgrep -af 'run_core_lt_rarebc_seed7_23.sh|rarebc_libero_core_lt_w3_tail9_seed7_b20|run_core_all_hf_download_then_apa_seed7_23.sh|core_all_hf_apa_seed7|run_core_lt_priority_after_apa_multiseed_23.sh' || true
    sleep 1800
  done
}

wait_gpus_23() {
  log "waiting for physical GPU 2/3 idle"
  python - <<'PY'
import subprocess, time
while True:
    txt = subprocess.check_output([
        'nvidia-smi',
        '--query-gpu=index,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits',
    ], text=True)
    rows = {}
    for line in txt.strip().splitlines():
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 3:
            rows[int(parts[0])] = (int(parts[1]), int(parts[2]))
    ok = all(i in rows and rows[i][0] < 2000 and rows[i][1] < 20 for i in (2, 3))
    print(time.strftime('%F %T'), rows, 'ok=', ok, flush=True)
    if ok:
        break
    time.sleep(300)
PY
}

setup_env() {
  source /opt/miniconda3/etc/profile.d/conda.sh
  conda activate "$ENV"
  cd "$ROOT"
  export CUDA_VISIBLE_DEVICES="$GPU_IDS"
  export EVAL_ALLOWED_GPUS="$GPU_IDS"
  export MUJOCO_GL=egl
  export PYOPENGL_PLATFORM=egl
  export HF_ENDPOINT=https://hf-mirror.com
  export HF_HOME=/mnt/data/cyh/.cache/huggingface
  export TRANSFORMERS_CACHE=/mnt/data/cyh/.cache/huggingface
  export TOKENIZERS_PARALLELISM=false
  export WANDB_DISABLED=true
  export PRISMATIC_DATA_ROOT="$ROOT/data/prismatic"
  export PYTHONPATH="$ROOT/LIBERO:$ROOT:${PYTHONPATH:-}"
  export PYTHONUNBUFFERED=1
  export HF_HUB_DISABLE_TELEMETRY=1
  mkdir -p "$PRISMATIC_DATA_ROOT" "$HF_HOME" "runs/$RUN_GROUP" "results/$RUN_GROUP"
  : > .hf_token
}

latest_step() {
  local run_root="$1"
  find "$run_root/checkpoints" -maxdepth 1 -type f -name 'step-*.pt' 2>/dev/null \
    | sed -E 's/.*step-0*([0-9]+)-.*/\1/' \
    | sort -n \
    | tail -1
}

link_eval_steps() {
  local run_root="$1"
  local step="$2"
  local ckpt_dir="$run_root/checkpoints"
  local ckpt_file ckpt_base suffix alias dest
  ckpt_file=$(find "$ckpt_dir" -maxdepth 1 -type f -name 'step-*.pt' | sort | tail -n 1)
  ckpt_base=$(basename "$ckpt_file")
  suffix="${ckpt_base#step-}"
  suffix="${suffix#*-}"
  for alias in "$step" "$(printf '%05d' "$step")" "$(printf '%06d' "$step")"; do
    dest="step-${alias}-${suffix}"
    [[ "$dest" == "$ckpt_base" ]] || ln -sf "$ckpt_base" "$ckpt_dir/$dest"
  done
}

has_eval_log() {
  local result_root="$1"
  find "$result_root" -type f -name '000.log' 2>/dev/null | grep -q .
}

require_apa_ready() {
  setup_env
  if [[ ! -d /mnt/data/cyh/tensorflow_datasets/libero_core_apa/1.0.0 ]]; then
    log "ERROR: libero_core_apa TFDS is missing; same-pipeline APA did not complete cleanly"
    exit 4
  fi
  python - <<'PY'
from prismatic.vla.datasets.rlds.oxe.configs import OXE_DATASET_CONFIGS
from prismatic.vla.datasets.rlds.oxe.mixtures import OXE_NAMED_MIXTURES
from prismatic.vla.datasets.rlds.oxe.transforms import OXE_STANDARDIZATION_TRANSFORMS
import tensorflow_datasets as tfds
name = 'libero_core_apa'
assert name in OXE_DATASET_CONFIGS, name
assert OXE_NAMED_MIXTURES.get(name), name
assert name in OXE_STANDARDIZATION_TRANSFORMS, name
builder = tfds.builder_from_directory('/mnt/data/cyh/tensorflow_datasets/libero_core_apa/1.0.0')
print('libero_core_apa ready', builder.info.splits)
PY
}

train_apa_rbtad() {
  setup_env
  local run_root="runs/$RUN_GROUP/$RUN_ID"
  local step
  step=$(latest_step "$run_root" || true)
  if [[ -n "${step:-}" ]]; then
    log "skip APA+RBTAD train; existing_step=$step run=$run_root"
    return 0
  fi
  wait_gpus_23
  setup_env
  log "train APA+RBTAD seed7 run=$RUN_ID"
  torchrun --nnodes 1 --nproc-per-node 2 --master_addr 127.0.0.1 --master_port 33197 vla_scripts/train.py \
    --pretrained_checkpoint "$PRETRAIN" \
    --vla.type "$VLA_TYPE" \
    --vla.data_mix "$DATA_MIX" \
    --vla.expected_world_size 2 \
    --vla.global_batch_size 20 \
    --vla.per_device_batch_size 10 \
    --data_root_dir /mnt/data/cyh/tensorflow_datasets \
    --run_root_dir "runs/$RUN_GROUP" \
    --run_id "$RUN_ID" \
    --save_interval 5000 \
    --seed 7 \
    --tcad_lambda 0.1 \
    --tcad_ratio 0.5 \
    --tcad_margin 0.2 \
    --tcad_tail_max_count 9 \
    --tcad_conf_gate batch_median \
    --tcad_negative_mode manifest \
    --rare_bc_max_count 9 \
    --rare_bc_weight 3.0
  tail -n 50 "$run_root/tcad-debug.csv" 2>/dev/null || true
  log "train APA+RBTAD done latest_step=$(latest_step "$run_root" || true)"
}

eval_apa_rbtad() {
  setup_env
  local run_root="runs/$RUN_GROUP/$RUN_ID"
  local step result_root save_root stamp
  step=$(latest_step "$run_root" || true)
  if [[ -z "${step:-}" ]]; then
    log "ERROR: no APA+RBTAD checkpoint found for eval"
    exit 5
  fi
  link_eval_steps "$run_root" "$step"
  result_root="results/$RUN_GROUP/$RUN_ID/$step"
  if has_eval_log "$result_root"; then
    log "skip APA+RBTAD eval; result exists under $result_root"
    return 0
  fi
  wait_gpus_23
  setup_env
  stamp=$(date +%Y%m%d_%H%M%S)
  save_root="$result_root/apa_rbtad_seed7_step${step}_30trials_egl_${stamp}"
  log "eval APA+RBTAD seed7 step=$step save_root=$save_root"
  python vla_scripts/parallel_libero_evaluator_egl.py \
    --num-trails-per-task 30 \
    --num-gpus 2 \
    --num-processes 10 \
    --task-suite-name "$TASK_SUITE" \
    --pretrained-checkpoint "$run_root" \
    --unnorm_key "$UNNORM_KEY" \
    --save-root "$save_root" \
    --steps "$step" \
    --instruction-formatting False
  grep -R -E 'Overall success rate|Task .*success rate' "$save_root" 2>/dev/null || true
  log "eval APA+RBTAD done"
}

{
  log "APA+RBTAD complementarity queue started"
  wait_for_priority_front
  require_apa_ready
  train_apa_rbtad
  eval_apa_rbtad
  cd "$ROOT"
  python code/summarize_core_lt_priority_results.py \
    --root "$ROOT" \
    --json-out autoresearch/state/core_lt_priority_summary.json \
    --md-out autoresearch/state/core_lt_priority_summary.md || true
  log "APA+RBTAD complementarity queue complete"
} >> "$LOG" 2>&1
