#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/data/cyh/VLA-long-tail
ENV=/mnt/data/cyh/envs/vla-long-tail
SEED=${SEED:-13}
SUITE=${SUITE:-libero_spatial}
DATA_MIX=${DATA_MIX:-libero_spatial_lt}
UNNORM_KEY=${UNNORM_KEY:-libero_spatial_lt}
LIMIT_STEPS=${LIMIT_STEPS:-1000}
CORRECT_STEPS=${CORRECT_STEPS:-100}
NUM_TRIALS=${NUM_TRIALS:-30}
ALPHA=${ALPHA:-0.5}
RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}
RUN_ROOT_BASE=runs/spatial_lt_multiseed
RUN_ROOT_BARC=runs/spatial_lt_barc_multiseed
RUN_ROOT_RSDF=runs/spatial_lt_rsdf_multiseed
RESULT_ROOT=results/spatial_lt_multiseed_confirm30
LOG=/mnt/data/cyh/spatial_lt_rsdf_seed${SEED}_confirm30_${RUN_STAMP}.log
BASE_ID="baseline_${DATA_MIX}_s${LIMIT_STEPS}_seed${SEED}_b20_${RUN_STAMP}"
BARC_ID="barc_${DATA_MIX}_base${LIMIT_STEPS}p${CORRECT_STEPS}_seed${SEED}_b20_a10p0_${RUN_STAMP}"
RSDF_ID="rsdf_visionllm_barc${CORRECT_STEPS}_seed${SEED}_a${ALPHA//./p}_${RUN_STAMP}"

wait_for_gpus() {
  echo "== waiting for GPUs 2/3 $(date -Is) =="
  while true; do
    local used2 used3 util2 util3
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

link_eval_steps() {
  local run_root="$1"
  local run_id="$2"
  local step="$3"
  local ckpt_file ckpt_base suffix alias dest
  ckpt_file=$(find "${run_root}/${run_id}/checkpoints" -maxdepth 1 -type f -name 'step-*.pt' | sort | tail -n 1)
  if [[ -z "${ckpt_file}" ]]; then
    echo "No checkpoint found for ${run_root}/${run_id}" >&2
    return 1
  fi
  ckpt_base=$(basename "${ckpt_file}")
  suffix="${ckpt_base#step-}"
  suffix="${suffix#*-}"
  for alias in "${step}" "$(printf "%04d" "${step}")" "$(printf "%05d" "${step}")" "$(printf "%06d" "${step}")"; do
    dest="step-${alias}-${suffix}"
    if [[ "${dest}" != "${ckpt_base}" ]]; then
      ln -sf "${ckpt_base}" "${run_root}/${run_id}/checkpoints/${dest}"
    fi
  done
}

run_eval() {
  local label="$1"
  local ckpt_dir="$2"
  local step="$3"
  local save_root="${RESULT_ROOT}/${label}/step${step}/${label}_${SUITE}_${NUM_TRIALS}trials_egl_$(date +%Y%m%d_%H%M%S)"
  echo "== eval start $(date -Is) label=${label} ckpt=${ckpt_dir} step=${step} save_root=${save_root} =="
  python vla_scripts/parallel_libero_evaluator_egl.py \
    --num-trails-per-task "${NUM_TRIALS}" \
    --num-gpus 2 \
    --num-processes 10 \
    --task-suite-name "${SUITE}" \
    --pretrained-checkpoint "${ckpt_dir}" \
    --unnorm_key "${UNNORM_KEY}" \
    --save-root "${save_root}" \
    --steps "${step}" \
    --instruction-formatting False
  echo "== eval done $(date -Is) label=${label} =="
  grep -R -E "Overall success rate|Task .*success rate" "${save_root}" 2>/dev/null || true
}

make_rsdf() {
  local base_run="$1"
  local corr_run="$2"
  local out_run="$3"
  mkdir -p "${out_run}/checkpoints"
  python - "${base_run}" "${corr_run}" "${out_run}" "${ALPHA}" <<'PY'
import sys, shutil, torch
from pathlib import Path
from collections import OrderedDict
base_run, corr_run, out_run, alpha = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
selected = {'vision_backbone', 'llm_backbone'}
base_ckpt = sorted((Path(base_run)/'checkpoints').glob('step-*.pt'))[-1]
corr_ckpt = sorted((Path(corr_run)/'checkpoints').glob('step-*.pt'))[-1]
out_run = Path(out_run)
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

{
  echo "== seed${SEED} RSDF confirm start $(date -Is) stamp=${RUN_STAMP} =="
  source /opt/miniconda3/etc/profile.d/conda.sh
  conda activate "${ENV}"
  cd "${ROOT}"
  export CUDA_VISIBLE_DEVICES=2,3
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
  unset TARGET_TASK_INSTRUCTION
  unset TARGET_TASK_WEIGHT
  mkdir -p "${PRISMATIC_DATA_ROOT}" "${HF_HOME}" "${RUN_ROOT_BASE}" "${RUN_ROOT_BARC}" "${RUN_ROOT_RSDF}" "${RESULT_ROOT}" autoresearch/state autoresearch/logs
  : > .hf_token

  wait_for_gpus
  echo "== gpu acquired $(date -Is) =="
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits

  echo "== baseline train start $(date -Is) run_id=${BASE_ID} =="
  torchrun --nnodes 1 --nproc-per-node 2 --master_addr 127.0.0.1 --master_port 29913 vla_scripts/train.py \
    --pretrained_checkpoint pretrained/minivla-libero90-prismatic/checkpoints/step-122500-epoch-55-loss=0.0743.pt \
    --vla.type "prism-qwen25-dinosiglip-224px+0_5b+mx-libero-core-lt" \
    --vla.data_mix "${DATA_MIX}" \
    --vla.expected_world_size 2 \
    --vla.global_batch_size 20 \
    --vla.per_device_batch_size 10 \
    --data_root_dir /mnt/data/cyh/tensorflow_datasets \
    --run_root_dir "${RUN_ROOT_BASE}" \
    --run_id "${BASE_ID}" \
    --save_interval "${LIMIT_STEPS}" \
    --seed "${SEED}" \
    --tcad_lambda 0.0 \
    --tcad_ratio 0.0 \
    --tcad_margin 0.2 \
    --tcad_tail_max_count 0 \
    --tcad_conf_gate none \
    --tcad_negative_mode manifest \
    --rare_bc_max_count 0 \
    --rare_bc_weight 1.0 \
    --train_limit_steps "${LIMIT_STEPS}"
  echo "== baseline train done $(date -Is) =="
  link_eval_steps "${RUN_ROOT_BASE}" "${BASE_ID}" "${LIMIT_STEPS}"
  base_ckpt=$(find "${RUN_ROOT_BASE}/${BASE_ID}/checkpoints" -maxdepth 1 -type f -name 'step-*.pt' | sort | tail -n 1)
  echo "base_ckpt=${base_ckpt}"

  echo "== BARC smoke start $(date -Is) run_id=smoke_${BARC_ID} =="
  torchrun --nnodes 1 --nproc-per-node 2 --master_addr 127.0.0.1 --master_port 29914 vla_scripts/train.py \
    --pretrained_checkpoint "${base_ckpt}" \
    --vla.type "prism-qwen25-dinosiglip-224px+0_5b+mx-libero-core-lt" \
    --vla.data_mix "${DATA_MIX}" \
    --vla.expected_world_size 2 \
    --vla.global_batch_size 20 \
    --vla.per_device_batch_size 10 \
    --data_root_dir /mnt/data/cyh/tensorflow_datasets \
    --run_root_dir "${RUN_ROOT_BARC}" \
    --run_id "smoke_${BARC_ID}" \
    --save_interval 5 \
    --seed "${SEED}" \
    --tcad_lambda 0.1 \
    --tcad_ratio 0.5 \
    --tcad_margin 0.2 \
    --tcad_tail_max_count 9 \
    --tcad_conf_gate batch_median \
    --tcad_negative_mode manifest \
    --rare_bc_max_count 9 \
    --rare_bc_weight 2.0 \
    --anchor_l2_lambda 10.0 \
    --anchor_l2_filter llm_backbone \
    --train_limit_steps 5
  tail -n 20 "${RUN_ROOT_BARC}/smoke_${BARC_ID}/tcad-debug.csv" 2>/dev/null || true

  echo "== BARC correction start $(date -Is) run_id=${BARC_ID} =="
  torchrun --nnodes 1 --nproc-per-node 2 --master_addr 127.0.0.1 --master_port 29915 vla_scripts/train.py \
    --pretrained_checkpoint "${base_ckpt}" \
    --vla.type "prism-qwen25-dinosiglip-224px+0_5b+mx-libero-core-lt" \
    --vla.data_mix "${DATA_MIX}" \
    --vla.expected_world_size 2 \
    --vla.global_batch_size 20 \
    --vla.per_device_batch_size 10 \
    --data_root_dir /mnt/data/cyh/tensorflow_datasets \
    --run_root_dir "${RUN_ROOT_BARC}" \
    --run_id "${BARC_ID}" \
    --save_interval "${CORRECT_STEPS}" \
    --seed "${SEED}" \
    --tcad_lambda 0.1 \
    --tcad_ratio 0.5 \
    --tcad_margin 0.2 \
    --tcad_tail_max_count 9 \
    --tcad_conf_gate batch_median \
    --tcad_negative_mode manifest \
    --rare_bc_max_count 9 \
    --rare_bc_weight 2.0 \
    --anchor_l2_lambda 10.0 \
    --anchor_l2_filter llm_backbone \
    --train_limit_steps "${CORRECT_STEPS}"
  echo "== BARC correction done $(date -Is) =="
  tail -n 20 "${RUN_ROOT_BARC}/${BARC_ID}/tcad-debug.csv" 2>/dev/null || true
  link_eval_steps "${RUN_ROOT_BARC}" "${BARC_ID}" "${CORRECT_STEPS}"

  echo "== RSDF build start $(date -Is) run_id=${RSDF_ID} =="
  make_rsdf "${RUN_ROOT_BASE}/${BASE_ID}" "${RUN_ROOT_BARC}/${BARC_ID}" "${RUN_ROOT_RSDF}/${RSDF_ID}"

  run_eval "${RSDF_ID}" "${RUN_ROOT_RSDF}/${RSDF_ID}" "${CORRECT_STEPS}"
  run_eval "${BASE_ID}" "${RUN_ROOT_BASE}/${BASE_ID}" "${LIMIT_STEPS}"
  echo "== seed${SEED} RSDF confirm all done $(date -Is) =="
} >> "${LOG}" 2>&1
