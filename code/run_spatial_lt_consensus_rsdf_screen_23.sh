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
RUN_ROOT=runs/spatial_lt_consensus_rsdf
RESULT_ROOT=results/spatial_lt_consensus_rsdf
LOG=/mnt/data/cyh/spatial_lt_consensus_rsdf_screen_${RUN_STAMP}.log

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

make_consensus_rsdf() {
  local target_base="$1" out_run="$2"
  mkdir -p "${out_run}/checkpoints"
  python - "${target_base}" "${out_run}" "${ALPHA}" <<'PY'
import json, shutil, sys, torch
from pathlib import Path
from collections import OrderedDict

target_base, out_run, alpha = sys.argv[1], Path(sys.argv[2]), float(sys.argv[3])
base7 = Path('runs/spatial_lt_screen/baseline_libero_spatial_lt_s1000_seed7_b20')
corr7 = Path('runs/spatial_lt_barc/barc_libero_spatial_lt_base1000p100_seed7_b20_a10p0')
base13 = Path('runs/spatial_lt_multiseed/baseline_libero_spatial_lt_s1000_seed13_b20_20260718_185539')
corr13 = Path('runs/spatial_lt_barc_multiseed/barc_libero_spatial_lt_base1000p100_seed13_b20_a10p0_20260718_185539')
selected = {'vision_backbone', 'llm_backbone'}

def latest(run):
    return sorted((Path(run) / 'checkpoints').glob('step-*.pt'))[-1]

target_ckpt, b7_ckpt, c7_ckpt, b13_ckpt, c13_ckpt = map(latest, [target_base, base7, corr7, base13, corr13])
print(f'target_ckpt={target_ckpt}')
print(f'b7_ckpt={b7_ckpt}')
print(f'c7_ckpt={c7_ckpt}')
print(f'b13_ckpt={b13_ckpt}')
print(f'c13_ckpt={c13_ckpt}')
print(f'selected={sorted(selected)} alpha={alpha} rule=sign-consensus-delta-average')

target = torch.load(target_ckpt, map_location='cpu')
b7 = torch.load(b7_ckpt, map_location='cpu')
c7 = torch.load(c7_ckpt, map_location='cpu')
b13 = torch.load(b13_ckpt, map_location='cpu')
c13 = torch.load(c13_ckpt, map_location='cpu')
out = {'model': OrderedDict()}
summary = {}
for module_name, target_sub in target['model'].items():
    if module_name not in selected:
        out['model'][module_name] = target_sub
        continue
    blended = OrderedDict()
    changed = total_elems = kept_elems = 0
    consensus_norm = applied_norm = 0.0
    for k, tv in target_sub.items():
        if not (torch.is_tensor(tv) and torch.is_floating_point(tv)):
            blended[k] = tv
            continue
        b7v, c7v = b7['model'][module_name].get(k), c7['model'][module_name].get(k)
        b13v, c13v = b13['model'][module_name].get(k), c13['model'][module_name].get(k)
        if not all(torch.is_tensor(x) and x.shape == tv.shape and torch.is_floating_point(x) for x in [b7v, c7v, b13v, c13v]):
            blended[k] = tv
            continue
        d7 = c7v.float() - b7v.float()
        d13 = c13v.float() - b13v.float()
        mask = (d7 * d13) > 0
        consensus = 0.5 * (d7 + d13) * mask
        blended[k] = (tv.float() + alpha * consensus).to(dtype=tv.dtype)
        changed += 1
        total_elems += mask.numel()
        kept_elems += int(mask.sum().item())
        consensus_norm += float(consensus.pow(2).sum())
        applied_norm += float((alpha * consensus).pow(2).sum())
    out['model'][module_name] = blended
    summary[module_name] = {
        'changed_tensors': changed,
        'kept_fraction': (kept_elems / total_elems) if total_elems else 0.0,
        'kept_elems': kept_elems,
        'total_elems': total_elems,
        'consensus_delta_norm': consensus_norm ** 0.5,
        'applied_delta_norm': applied_norm ** 0.5,
    }
    print(f"module={module_name} changed={changed} kept_fraction={summary[module_name]['kept_fraction']:.6f} consensus_delta_norm={summary[module_name]['consensus_delta_norm']:.6f} applied_delta_norm={summary[module_name]['applied_delta_norm']:.6f}")
for name in ['config.json', 'config.yaml', 'dataset_statistics.json']:
    src = Path(target_base) / name
    if src.exists():
        shutil.copy2(src, out_run / name)
(out_run / 'consensus_rsdf_manifest.json').write_text(json.dumps({
    'target_base': str(target_ckpt),
    'base7': str(b7_ckpt),
    'corr7': str(c7_ckpt),
    'base13': str(b13_ckpt),
    'corr13': str(c13_ckpt),
    'alpha': alpha,
    'selected': sorted(selected),
    'rule': 'apply average BARC delta only where seed7 and seed13 delta signs agree',
    'summary': summary,
}, indent=2), encoding='utf-8')
torch.save(out, out_run / 'checkpoints' / 'step-00100-epoch-00-loss=0.0000.pt')
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
  local seed="$1" base_run="$2" out_id
  out_id="consensus_rsdf_seed${seed}_a${ALPHA//./p}_${RUN_STAMP}"
  [[ -d "${base_run}" ]] || { echo "missing base_run=${base_run}" >&2; return 2; }
  echo "== build Consensus-Signed RSDF seed=${seed} =="
  make_consensus_rsdf "${base_run}" "${RUN_ROOT}/${out_id}"
  run_eval "${out_id}" "${RUN_ROOT}/${out_id}" 100
}

{
  echo "== Consensus-Signed RSDF screen start $(date -Is) stamp=${RUN_STAMP} =="
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
  mkdir -p "${RUN_ROOT}" "${RESULT_ROOT}"
  : > .hf_token
  wait_for_gpus
  python /mnt/data/cyh/patch_eval_fixed_init_ids.py
  python - <<'PY'
from pathlib import Path
p = Path('vla_scripts/parallel_libero_evaluator_egl.py')
compile(p.read_text(encoding='utf-8'), str(p), 'exec')
print('compile-ok', p)
PY
  run_seed 7 runs/spatial_lt_screen/baseline_libero_spatial_lt_s1000_seed7_b20
  run_seed 13 runs/spatial_lt_multiseed/baseline_libero_spatial_lt_s1000_seed13_b20_20260718_185539
  echo "== Consensus-Signed RSDF screen all done $(date -Is) =="
} >> "${LOG}" 2>&1