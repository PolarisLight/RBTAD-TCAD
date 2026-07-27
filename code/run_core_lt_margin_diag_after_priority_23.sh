#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/data/cyh/VLA-long-tail
ENV=/mnt/data/cyh/envs/vla-long-tail
LOG=/mnt/data/cyh/core_lt_margin_diag_after_priority_20260728.log
DIAG_SCRIPT="$ROOT/code/diagnose_core_lt_margins.py"
OUT_DIR="$ROOT/autoresearch/state/margin_diagnostics"

log() { echo "[$(date -Is)] $*"; }

wait_front_and_priority_done() {
  log "waiting for Rare-BC, APA, post-APA priority, and APA+RBTAD queues to finish"
  while pgrep -af 'run_core_lt_rarebc_seed7_23.sh|rarebc_libero_core_lt_w3_tail9_seed7_b20|run_core_all_hf_download_then_apa_seed7_23.sh|core_all_hf_apa_seed7|run_core_lt_priority_after_apa_multiseed_23.sh|run_core_lt_apa_rbtad_after_priority_23.sh|apa_rbtad_libero_core_apa_w3_tail9_confmedian_seed7_b20' >/dev/null; do
    pgrep -af 'run_core_lt_rarebc_seed7_23.sh|rarebc_libero_core_lt_w3_tail9_seed7_b20|run_core_all_hf_download_then_apa_seed7_23.sh|core_all_hf_apa_seed7|run_core_lt_priority_after_apa_multiseed_23.sh|run_core_lt_apa_rbtad_after_priority_23.sh|apa_rbtad_libero_core_apa_w3_tail9_confmedian_seed7_b20' || true
    sleep 1800
  done
}

wait_gpu2_idle() {
  log "waiting for physical GPU 2 idle"
  python - <<'PY'
import subprocess, time
while True:
    txt = subprocess.check_output([
        'nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu', '--format=csv,noheader,nounits'
    ], text=True)
    rows = {}
    for line in txt.strip().splitlines():
        idx, mem, util = [int(x.strip()) for x in line.split(',')[:3]]
        rows[idx] = (mem, util)
    ok = 2 in rows and rows[2][0] < 2000 and rows[2][1] < 20
    print(time.strftime('%F %T'), rows, 'gpu2_ok=', ok, flush=True)
    if ok:
        break
    time.sleep(300)
PY
}

latest_ckpt() {
  local run_root="$1"
  find "$run_root/checkpoints" -maxdepth 1 -type f -name 'step-*.pt' 2>/dev/null \
    | sed -E 's/.*step-0*([0-9]+)-.*/\1 &/' \
    | sort -n \
    | tail -1 \
    | cut -d' ' -f2-
}

add_model_arg_if_ckpt() {
  local label="$1"
  local run_root="$2"
  local ckpt
  ckpt=$(latest_ckpt "$run_root" || true)
  if [[ -n "${ckpt:-}" ]]; then
    MODEL_ARGS+=(--model "${label}=${ckpt}")
    log "diagnostic model ${label} -> ${ckpt}"
  else
    log "skip diagnostic model ${label}: no checkpoint under ${run_root}"
  fi
}

{
  log "Core-LT margin diagnostic watcher started"
  wait_front_and_priority_done
  wait_gpu2_idle

  source /opt/miniconda3/etc/profile.d/conda.sh
  conda activate "$ENV"
  cd "$ROOT"
  export CUDA_VISIBLE_DEVICES=2
  export MUJOCO_GL=egl
  export HF_ENDPOINT=https://hf-mirror.com
  export HF_HOME=/mnt/data/cyh/.cache/huggingface
  export TOKENIZERS_PARALLELISM=false
  export WANDB_DISABLED=true
  export PYTHONPATH="$ROOT/LIBERO:$ROOT:${PYTHONPATH:-}"
  mkdir -p "$OUT_DIR"
  : > .hf_token

  python code/summarize_core_lt_priority_results.py \
    --root "$ROOT" \
    --json-out "$ROOT/autoresearch/state/core_lt_priority_summary.json" \
    --md-out "$ROOT/autoresearch/state/core_lt_priority_summary.md"

  MODEL_ARGS=()
  add_model_arg_if_ckpt bc_seed7 "$ROOT/runs/miniVLA_libero_core_lt/prism-qwen25-dinosiglip-224px+0_5b+mx-libero-core-lt+n0+b10+x7"
  add_model_arg_if_ckpt rbtad_seed7 "$ROOT/runs/rbtad_lt_main/rbtad_w3_tail9_confmedian_seed7_b20"
  add_model_arg_if_ckpt rarebc_seed7 "$ROOT/runs/core_lt_ablation/rarebc_libero_core_lt_w3_tail9_seed7_b20"
  add_model_arg_if_ckpt matched_bc_seed7 "$ROOT/runs/core_lt_ablation/matchedbc_libero_core_lt_seed7_b20"
  add_model_arg_if_ckpt bc_seed13 "$ROOT/runs/core_lt_multiseed/baseline_libero_core_lt_seed13_b20"
  add_model_arg_if_ckpt rbtad_seed13 "$ROOT/runs/core_lt_multiseed/rbtad_libero_core_lt_w3_tail9_confmedian_seed13_b20"
  add_model_arg_if_ckpt bc_seed21 "$ROOT/runs/core_lt_multiseed/baseline_libero_core_lt_seed21_b20"
  add_model_arg_if_ckpt rbtad_seed21 "$ROOT/runs/core_lt_multiseed/rbtad_libero_core_lt_w3_tail9_confmedian_seed21_b20"
  add_model_arg_if_ckpt apa_seed7 "$ROOT/runs/core_lt_priority/apa_libero_core_apa_seed7_b20"
  add_model_arg_if_ckpt apa_rbtad_seed7 "$ROOT/runs/core_lt_complementarity/apa_rbtad_libero_core_apa_w3_tail9_confmedian_seed7_b20"

  if [[ ${#MODEL_ARGS[@]} -lt 4 ]]; then
    log "not enough checkpoints for margin diagnostic; args=${MODEL_ARGS[*]:-<none>}"
    exit 0
  fi

  stamp=$(date +%Y%m%d_%H%M%S)
  python "$DIAG_SCRIPT" \
    --tfds-dir /mnt/data/cyh/tensorflow_datasets/libero_core_lt/1.0.0 \
    --summary-json "$ROOT/autoresearch/state/core_lt_priority_summary.json" \
    --output "$OUT_DIR/core_lt_margin_diag_${stamp}.json" \
    --max-samples-per-task 40 \
    --max-episodes 180 \
    --tail-max-count 9 \
    --token-scope all \
    "${MODEL_ARGS[@]}"
  log "Core-LT margin diagnostic complete"
} >> "$LOG" 2>&1
