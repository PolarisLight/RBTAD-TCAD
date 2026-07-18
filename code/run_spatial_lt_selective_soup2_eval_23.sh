#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/data/cyh/VLA-long-tail
ENV=/mnt/data/cyh/envs/vla-long-tail
SEED=${SEED:-7}
SUITE=${SUITE:-libero_spatial}
UNNORM_KEY=${UNNORM_KEY:-libero_spatial_lt}
NUM_TRIALS=${NUM_TRIALS:-10}
ALPHA=${ALPHA:-0.5}
STEP=${STEP:-100}
PROBE_NAME=${PROBE_NAME:-spatial_lt_selective_soup2}
BASE_RUN=${BASE_RUN:-runs/spatial_lt_screen/baseline_libero_spatial_lt_s1000_seed7_b20}
CORR_RUN=${CORR_RUN:-runs/spatial_lt_barc/barc_libero_spatial_lt_base1000p100_seed7_b20_a10p0}
RUN_ROOT=runs/spatial_lt_selective_soup
RESULT_ROOT=results/spatial_lt_selective_soup
LOG=/mnt/data/cyh/${PROBE_NAME}_barc100_a${ALPHA//./p}_t${NUM_TRIALS}.log

make_soup() {
  local run_id="$1"
  local modules="$2"
  local ckpt_name="step-00100-epoch-00-loss=0.0000.pt"
  if [[ -f "${RUN_ROOT}/${run_id}/checkpoints/${ckpt_name}" ]]; then
    echo "soup exists: ${RUN_ROOT}/${run_id}/checkpoints/${ckpt_name}"
    return 0
  fi
  mkdir -p "${RUN_ROOT}/${run_id}/checkpoints"
  python - "$BASE_RUN" "$CORR_RUN" "${RUN_ROOT}/${run_id}" "$modules" "$ALPHA" <<'PY'
import sys, shutil, torch
from pathlib import Path
from collections import OrderedDict
base_run, corr_run, out_run, modules, alpha = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], float(sys.argv[5])
selected = set(m.strip() for m in modules.split(',') if m.strip())
base_ckpt = sorted((Path(base_run)/'checkpoints').glob('step-*.pt'))[-1]
corr_ckpt = sorted((Path(corr_run)/'checkpoints').glob('step-*.pt'))[-1]
out_run = Path(out_run)
out_run.mkdir(parents=True, exist_ok=True)
(out_run/'checkpoints').mkdir(exist_ok=True)
print(f'base_ckpt={base_ckpt}')
print(f'corr_ckpt={corr_ckpt}')
print(f'selected={sorted(selected)} alpha={alpha}')
base = torch.load(base_ckpt, map_location='cpu')
corr = torch.load(corr_ckpt, map_location='cpu')
out = {'model': OrderedDict()}
for module_name, base_sub in base['model'].items():
    corr_sub = corr['model'][module_name]
    if module_name not in selected:
        out['model'][module_name] = base_sub
        continue
    blended = OrderedDict()
    changed = 0
    for k, bv in base_sub.items():
        cv = corr_sub[k]
        if torch.is_tensor(bv) and torch.is_tensor(cv) and bv.shape == cv.shape and torch.is_floating_point(bv):
            blended[k] = (bv.float().mul(1.0 - alpha).add(cv.float(), alpha=alpha)).to(dtype=bv.dtype)
            changed += 1
        else:
            blended[k] = bv
    out['model'][module_name] = blended
    print(f'blended {module_name}: {changed}/{len(base_sub)} tensors')
for name in ['config.json', 'config.yaml', 'dataset_statistics.json']:
    src = Path(corr_run)/name
    if not src.exists():
        src = Path(base_run)/name
    if src.exists():
        shutil.copy2(src, out_run/name)
torch.save(out, out_run/'checkpoints'/'step-00100-epoch-00-loss=0.0000.pt')
print(f'wrote={out_run / "checkpoints" / "step-00100-epoch-00-loss=0.0000.pt"}')
PY
}

run_eval() {
  local run_id="$1"
  local label="$2"
  local stamp save_root
  stamp=$(date +%Y%m%d_%H%M%S)
  save_root="${RESULT_ROOT}/${run_id}/step${STEP}/${label}_${SUITE}_${NUM_TRIALS}trials_egl_${stamp}"
  echo "== eval start $(date -Is) run_id=${run_id} save_root=${save_root} =="
  python vla_scripts/parallel_libero_evaluator_egl.py \
    --num-trails-per-task "${NUM_TRIALS}" \
    --num-gpus 2 \
    --num-processes 10 \
    --task-suite-name "${SUITE}" \
    --pretrained-checkpoint "${RUN_ROOT}/${run_id}" \
    --unnorm_key "${UNNORM_KEY}" \
    --save-root "${save_root}" \
    --steps "${STEP}" \
    --instruction-formatting False
  echo "== eval done $(date -Is) run_id=${run_id} =="
  grep -R -E "Overall success rate|Task .*success rate" "${save_root}" 2>/dev/null || true
}

{
  echo "== ${PROBE_NAME} start $(date -Is) =="
  source /opt/miniconda3/etc/profile.d/conda.sh
  conda activate "$ENV"
  cd "$ROOT"
  export CUDA_VISIBLE_DEVICES=2,3
  export HF_ENDPOINT=https://hf-mirror.com
  export HF_HOME=/mnt/data/cyh/.cache/huggingface
  export TRANSFORMERS_CACHE=/mnt/data/cyh/.cache/huggingface
  export TOKENIZERS_PARALLELISM=false
  export WANDB_DISABLED=true
  export PRISMATIC_DATA_ROOT="$ROOT/data/prismatic"
  export PYTHONPATH="$ROOT/LIBERO:$ROOT:${PYTHONPATH:-}"
  export MUJOCO_GL=egl
  export PYTHONUNBUFFERED=1
  export TFDS_DATA_DIR=/mnt/data/cyh/tensorflow_datasets
  mkdir -p "$RUN_ROOT" "$RESULT_ROOT" "$PRISMATIC_DATA_ROOT" "$HF_HOME"
  : > .hf_token
  echo "== gpu precheck $(date -Is) =="
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
  used2=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 2 | tr -dc '0-9')
  used3=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3 | tr -dc '0-9')
  if [[ "${used2:-999999}" -gt 2000 || "${used3:-999999}" -gt 2000 ]]; then
    echo "GPU 2/3 not free enough: gpu2=${used2}MiB gpu3=${used3}MiB"
    exit 3
  fi

  make_soup "rsdf_barc100_llmonly_a${ALPHA//./p}" "llm_backbone"
  make_soup "rsdf_barc100_visionllm_a${ALPHA//./p}" "vision_backbone,llm_backbone"
  run_eval "rsdf_barc100_llmonly_a${ALPHA//./p}" "rsdf_llmonly"
  run_eval "rsdf_barc100_visionllm_a${ALPHA//./p}" "rsdf_visionllm"
  echo "== ${PROBE_NAME} all done $(date -Is) =="
} >> "$LOG" 2>&1