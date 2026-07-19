#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/data/cyh/VLA-long-tail
ENV=/mnt/data/cyh/envs/vla-long-tail
SUITE=${SUITE:-libero_spatial}
UNNORM_KEY=${UNNORM_KEY:-libero_spatial_lt}
NUM_TRIALS=${NUM_TRIALS:-10}
ALPHA=${ALPHA:-0.5}
EVAL_INIT_IDS=${EVAL_INIT_IDS:-0,1,2,3,4,5,6,7,8,9}
RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}
RUN_ROOT_AHP=runs/spatial_lt_ahp_rsdf
RESULT_ROOT=results/spatial_lt_ahp_rsdf
LOG=/mnt/data/cyh/spatial_lt_ahp_rsdf_screen_${RUN_STAMP}.log

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

make_ahp_rsdf() {
  local base_run="$1" corr_run="$2" out_run="$3"
  mkdir -p "${out_run}/checkpoints"
  python - "${base_run}" "${corr_run}" "${out_run}" "${ALPHA}" <<'PY'
import sys, shutil, torch
from pathlib import Path
from collections import OrderedDict
base_run, corr_run, out_run, alpha = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
selected = {'vision_backbone', 'llm_backbone'}
protected = {'llm.model.embed_tokens.weight', 'llm.lm_head.weight'}
base_ckpt = sorted((Path(base_run)/'checkpoints').glob('step-*.pt'))[-1]
corr_ckpt = sorted((Path(corr_run)/'checkpoints').glob('step-*.pt'))[-1]
out_run = Path(out_run)
print(f'base_ckpt={base_ckpt}')
print(f'corr_ckpt={corr_ckpt}')
print(f'selected={sorted(selected)} protected={sorted(protected)} alpha={alpha}')
base = torch.load(base_ckpt, map_location='cpu')
corr = torch.load(corr_ckpt, map_location='cpu')
out = {'model': OrderedDict()}
summary = {}
for module_name, base_sub in base['model'].items():
    corr_sub = corr['model'][module_name]
    if module_name not in selected:
        out['model'][module_name] = base_sub
        continue
    blended = OrderedDict()
    changed = protected_count = 0
    delta_norm = kept_delta_norm = 0.0
    for k, bv in base_sub.items():
        cv = corr_sub[k]
        if k in protected:
            blended[k] = bv
            protected_count += 1
            continue
        if torch.is_tensor(bv) and torch.is_tensor(cv) and bv.shape == cv.shape and torch.is_floating_point(bv):
            delta = cv.float() - bv.float()
            delta_norm += float(delta.pow(2).sum())
            blended[k] = (bv.float().add(delta, alpha=alpha)).to(dtype=bv.dtype)
            kept_delta_norm += float((alpha * delta).pow(2).sum())
            changed += 1
        else:
            blended[k] = bv
    out['model'][module_name] = blended
    summary[module_name] = dict(changed=changed, protected=protected_count, delta_norm=delta_norm ** 0.5, kept_delta_norm=kept_delta_norm ** 0.5)
    print(f"module={module_name} changed={changed} protected={protected_count} delta_norm={summary[module_name]['delta_norm']:.6f} kept_delta_norm={summary[module_name]['kept_delta_norm']:.6f}")
for name in ['config.json', 'config.yaml', 'dataset_statistics.json']:
    src = Path(corr_run)/name
    if not src.exists():
        src = Path(base_run)/name
    if src.exists():
        shutil.copy2(src, out_run/name)
(out_run/'ahp_rsdf_manifest.json').write_text(__import__('json').dumps(dict(base=str(base_ckpt), corr=str(corr_ckpt), alpha=alpha, selected=sorted(selected), protected=sorted(protected), summary=summary), indent=2), encoding='utf-8')
torch.save(out, out_run/'checkpoints'/'step-00100-epoch-00-loss=0.0000.pt')
print(f'wrote={out_run / "checkpoints" / "step-00100-epoch-00-loss=0.0000.pt"}')
PY
}

run_eval() {
  local label="$1" checkpoint="$2" step="$3"
  local save_root="${RESULT_ROOT}/${label}/step${step}/${label}_${SUITE}_${NUM_TRIALS}trials_initfixed_${RUN_STAMP}"
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

run_seed() {
  local seed="$1" base_run="$2" corr_run="$3" out_id
  out_id="ahp_rsdf_visionllm_seed${seed}_a${ALPHA//./p}_${RUN_STAMP}"
  [[ -d "${base_run}" ]] || { echo "missing base_run=${base_run}" >&2; return 2; }
  [[ -d "${corr_run}" ]] || { echo "missing corr_run=${corr_run}" >&2; return 3; }
  echo "== build AHP-RSDF seed=${seed} =="
  make_ahp_rsdf "${base_run}" "${corr_run}" "${RUN_ROOT_AHP}/${out_id}"
  run_eval "baseline_seed${seed}" "${base_run}" 1000
  run_eval "${out_id}" "${RUN_ROOT_AHP}/${out_id}" 100
}

{
  echo "== AHP-RSDF screen start $(date -Is) stamp=${RUN_STAMP} =="
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
  mkdir -p "${RUN_ROOT_AHP}" "${RESULT_ROOT}"
  : > .hf_token
  wait_for_gpus
  python /mnt/data/cyh/patch_eval_fixed_init_ids.py
  python - <<'PY'
from pathlib import Path
p = Path('vla_scripts/parallel_libero_evaluator_egl.py')
compile(p.read_text(encoding='utf-8'), str(p), 'exec')
print('compile-ok', p)
PY
  run_seed 7 runs/spatial_lt_screen/baseline_libero_spatial_lt_s1000_seed7_b20 runs/spatial_lt_barc/barc_libero_spatial_lt_base1000p100_seed7_b20_a10p0
  run_seed 13 runs/spatial_lt_multiseed/baseline_libero_spatial_lt_s1000_seed13_b20_20260718_185539 runs/spatial_lt_barc_multiseed/barc_libero_spatial_lt_base1000p100_seed13_b20_a10p0_20260718_185539
  echo "== AHP-RSDF screen all done $(date -Is) =="
} >> "${LOG}" 2>&1