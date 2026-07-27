#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/data/cyh/VLA-long-tail
ENV=/mnt/data/cyh/envs/vla-long-tail
LOG=/mnt/data/cyh/core_lt_priority_after_apa_multiseed_20260728.log
GPU_IDS=2,3
PRETRAIN=pretrained/minivla-libero90-prismatic/checkpoints/step-122500-epoch-55-loss=0.0743.pt
VLA_TYPE=prism-qwen25-dinosiglip-224px+0_5b+mx-libero-core-lt
DATA_MIX=libero_core_lt
TASK_SUITE=libero_core
UNNORM_KEY=libero_core_lt

log() { echo "[$(date -Is)] $*"; }
active_pgrep() {
  pgrep -af "$1" | grep -v 'bash -c bash -n' | grep -v 'bash -c .*nohup /mnt/data/cyh/run_' | grep -v 'pgrep -af'
}

wait_for_front_queue() {
  log "waiting for Rare-BC + same-pipeline APA front queue to finish"
  while active_pgrep 'run_core_lt_rarebc_seed7_23.sh|rarebc_libero_core_lt_w3_tail9_seed7_b20|run_core_all_hf_download_then_apa_seed7_23.sh|core_all_hf_apa_seed7' >/dev/null; do
    active_pgrep 'run_core_lt_rarebc_seed7_23.sh|rarebc_libero_core_lt_w3_tail9_seed7_b20|run_core_all_hf_download_then_apa_seed7_23.sh|core_all_hf_apa_seed7' || true
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
        '--format=csv,noheader,nounits'
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
  export MUJOCO_GL=egl
  export HF_ENDPOINT=https://hf-mirror.com
  export HF_HOME=/mnt/data/cyh/.cache/huggingface
  export TOKENIZERS_PARALLELISM=false
  export WANDB_DISABLED=true
  export PRISMATIC_DATA_ROOT="$ROOT/data/prismatic"
  export PYTHONPATH="$ROOT/LIBERO:$ROOT:${PYTHONPATH:-}"
  mkdir -p "$PRISMATIC_DATA_ROOT" "$HF_HOME"
  : > .hf_token
}

latest_step() {
  local run_root="$1"
  find "$run_root/checkpoints" -maxdepth 1 -type f -name 'step-*.pt' 2>/dev/null \
    | sed -E 's/.*step-0*([0-9]+)-.*/\1/' \
    | sort -n \
    | tail -1
}

has_eval_log() {
  local result_root="$1"
  find "$result_root" -type f -name '000.log' 2>/dev/null | grep -q .
}

train_job() {
  local kind="$1"
  local seed="$2"
  local run_group="$3"
  local run_id="$4"
  local run_root="runs/$run_group/$run_id"
  local step=""

  setup_env
  mkdir -p "runs/$run_group"
  step=$(latest_step "$run_root" || true)
  if [[ -n "${step:-}" ]]; then
    log "skip train kind=$kind seed=$seed existing_step=$step run=$run_root"
    return 0
  fi

  wait_gpus_23
  setup_env
  local port=$((33000 + seed + (${#kind} * 17)))
  log "train start kind=$kind seed=$seed run=$run_id port=$port"

  if [[ "$kind" == "bc" ]]; then
    torchrun --nnodes 1 --nproc-per-node 2 --master_addr 127.0.0.1 --master_port "$port" vla_scripts/train.py \
      --pretrained_checkpoint "$PRETRAIN" \
      --vla.type "$VLA_TYPE" \
      --vla.data_mix "$DATA_MIX" \
      --vla.expected_world_size 2 \
      --vla.global_batch_size 20 \
      --vla.per_device_batch_size 10 \
      --data_root_dir /mnt/data/cyh/tensorflow_datasets \
      --run_root_dir "runs/$run_group" \
      --run_id "$run_id" \
      --save_interval 5000 \
      --seed "$seed" \
      --tcad_lambda 0.0 \
      --tcad_ratio 0.0 \
      --rare_bc_max_count 0 \
      --rare_bc_weight 1.0
  elif [[ "$kind" == "rbtad" ]]; then
    torchrun --nnodes 1 --nproc-per-node 2 --master_addr 127.0.0.1 --master_port "$port" vla_scripts/train.py \
      --pretrained_checkpoint "$PRETRAIN" \
      --vla.type "$VLA_TYPE" \
      --vla.data_mix "$DATA_MIX" \
      --vla.expected_world_size 2 \
      --vla.global_batch_size 20 \
      --vla.per_device_batch_size 10 \
      --data_root_dir /mnt/data/cyh/tensorflow_datasets \
      --run_root_dir "runs/$run_group" \
      --run_id "$run_id" \
      --save_interval 5000 \
      --seed "$seed" \
      --tcad_lambda 0.1 \
      --tcad_ratio 0.5 \
      --tcad_margin 0.2 \
      --tcad_tail_max_count 9 \
      --tcad_conf_gate batch_median \
      --rare_bc_max_count 9 \
      --rare_bc_weight 3.0
  else
    log "unknown kind=$kind"
    exit 2
  fi
  log "train done kind=$kind seed=$seed run=$run_id latest_step=$(latest_step "$run_root" || true)"
}

eval_job() {
  local kind="$1"
  local seed="$2"
  local run_group="$3"
  local run_id="$4"
  local run_root="runs/$run_group/$run_id"
  local step
  step=$(latest_step "$run_root" || true)
  if [[ -z "${step:-}" ]]; then
    log "no checkpoint for eval kind=$kind seed=$seed run=$run_root"
    exit 3
  fi
  local result_root="results/$run_group/$run_id/$step"
  if has_eval_log "$result_root"; then
    log "skip eval kind=$kind seed=$seed existing_result_root=$result_root"
    return 0
  fi
  wait_gpus_23
  setup_env
  local stamp save_root
  stamp=$(date +%Y%m%d_%H%M%S)
  save_root="$result_root/${kind}_seed${seed}_step${step}_30trials_egl_${stamp}"
  log "eval start kind=$kind seed=$seed step=$step save_root=$save_root"
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
  log "eval done kind=$kind seed=$seed step=$step"
}

run_pair() {
  local kind="$1"
  local seed="$2"
  local run_group="$3"
  local run_id="$4"
  train_job "$kind" "$seed" "$run_group" "$run_id"
  eval_job "$kind" "$seed" "$run_group" "$run_id"
}

{
  log "priority-after-APA multiseed queue started"
  wait_for_front_queue
  log "front queue finished; starting priority continuation"
  run_pair bc 7 core_lt_ablation matchedbc_libero_core_lt_seed7_b20
  run_pair bc 13 core_lt_multiseed baseline_libero_core_lt_seed13_b20
  run_pair rbtad 13 core_lt_multiseed rbtad_libero_core_lt_w3_tail9_confmedian_seed13_b20
  run_pair bc 21 core_lt_multiseed baseline_libero_core_lt_seed21_b20
  run_pair rbtad 21 core_lt_multiseed rbtad_libero_core_lt_w3_tail9_confmedian_seed21_b20
  log "priority-after-APA multiseed queue complete"
} >> "$LOG" 2>&1