#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/data/cyh/VLA-long-tail
ENV=/mnt/data/cyh/envs/vla-long-tail
DATA_ROOT=/mnt/data/cyh/tensorflow_datasets
LOG=/mnt/data/cyh/core_all_hf_apa_seed7_20260728.log
HF_BASE=https://hf-mirror.com/datasets/yifengzhu-hf/LIBERO-datasets/resolve/main
RUN_STAMP=${RUN_STAMP:-20260728_apa_fullraw_seed7}
RUN_ROOT=runs/core_lt_priority
RESULT_ROOT=results/core_lt_priority

OBJECT_FILES=(
  pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5
  pick_up_the_bbq_sauce_and_place_it_in_the_basket_demo.hdf5
  pick_up_the_butter_and_place_it_in_the_basket_demo.hdf5
  pick_up_the_chocolate_pudding_and_place_it_in_the_basket_demo.hdf5
  pick_up_the_cream_cheese_and_place_it_in_the_basket_demo.hdf5
  pick_up_the_ketchup_and_place_it_in_the_basket_demo.hdf5
  pick_up_the_milk_and_place_it_in_the_basket_demo.hdf5
  pick_up_the_orange_juice_and_place_it_in_the_basket_demo.hdf5
  pick_up_the_salad_dressing_and_place_it_in_the_basket_demo.hdf5
  pick_up_the_tomato_sauce_and_place_it_in_the_basket_demo.hdf5
)

GOAL_FILES=(
  open_the_middle_drawer_of_the_cabinet_demo.hdf5
  open_the_top_drawer_and_put_the_bowl_inside_demo.hdf5
  push_the_plate_to_the_front_of_the_stove_demo.hdf5
  put_the_bowl_on_the_plate_demo.hdf5
  put_the_bowl_on_the_stove_demo.hdf5
  put_the_bowl_on_top_of_the_cabinet_demo.hdf5
  put_the_cream_cheese_in_the_bowl_demo.hdf5
  put_the_wine_bottle_on_the_rack_demo.hdf5
  put_the_wine_bottle_on_top_of_the_cabinet_demo.hdf5
  turn_on_the_stove_demo.hdf5
)

download_one() {
  local suite="$1"
  local file="$2"
  local out_dir="${ROOT}/libero_raw/${suite}"
  mkdir -p "${out_dir}"
  if [[ -s "${out_dir}/${file}" ]]; then
    echo "== exists ${suite}/${file} $(stat -c%s "${out_dir}/${file}") bytes =="
    return 0
  fi
  echo "== download ${suite}/${file} $(date -Is) =="
  wget -c --tries=20 --timeout=60 --read-timeout=60 --progress=dot:giga \
    "${HF_BASE}/${suite}/${file}" \
    -O "${out_dir}/${file}"
}

wait_for_gpus() {
  echo "== wait for GPUs 2/3 $(date -Is) =="
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

wait_for_old_apa() {
  echo "== wait for old APA generator/pipeline to exit $(date -Is) =="
  while pgrep -af "run_core_lt_apa_priority_23.sh|parallel_libero_dataset_regenerator.py --num-gpus 2" >/dev/null; do
    pgrep -af "run_core_lt_apa_priority_23.sh|parallel_libero_dataset_regenerator.py --num-gpus 2" || true
    sleep 900
  done
}

wait_for_rarebc_done() {
  echo "== wait for Rare-BC ablation to finish before APA GPU work $(date -Is) =="
  while pgrep -af "run_core_lt_rarebc_seed7_23.sh|rarebc_libero_core_lt_w3_tail9_seed7_b20|parallel_libero_evaluator_egl.py.*rarebc_seed7" >/dev/null; do
    pgrep -af "run_core_lt_rarebc_seed7_23.sh|rarebc_libero_core_lt_w3_tail9_seed7_b20|parallel_libero_evaluator_egl.py.*rarebc_seed7" || true
    sleep 900
  done
}
count_hdf5() {
  local dir="$1"
  find "${dir}" -maxdepth 1 -type f -name "*.hdf5" 2>/dev/null | wc -l
}

regenerate_suite_full() {
  local suite="$1"
  local expected="$2"
  local raw_dir="${ROOT}/libero_raw/${suite}"
  local official_dir="${ROOT}/dataset_all/${suite}_no_noops"
  local scratch_dir="${ROOT}/dataset_all/${suite}_no_noops_fullraw_${RUN_STAMP}"
  local count

  count=$(count_hdf5 "${official_dir}")
  if [[ "${count}" -ge "${expected}" ]]; then
    echo "== ${suite} official no-noops complete: ${count}/${expected} =="
    return 0
  fi

  count=$(count_hdf5 "${scratch_dir}")
  if [[ "${count}" -lt "${expected}" ]]; then
    echo "== regenerate ${suite} to scratch ${scratch_dir} $(date -Is) =="
    wait_for_gpus
    python scripts/dataset/parallel_libero_dataset_regenerator.py \
      --num-gpus 2 \
      --libero-task-suite "${suite}" \
      --libero-raw-data-dir "${raw_dir}" \
      --libero-target-dir "${scratch_dir}"
  fi

  count=$(count_hdf5 "${scratch_dir}")
  if [[ "${count}" -lt "${expected}" ]]; then
    echo "ERROR: ${suite} scratch no-noops incomplete: ${count}/${expected}" >&2
    return 3
  fi

  mkdir -p "${official_dir}"
  cp -n "${scratch_dir}"/*.hdf5 "${official_dir}/"
  count=$(count_hdf5 "${official_dir}")
  echo "== ${suite} official no-noops after fill: ${count}/${expected} =="
  [[ "${count}" -ge "${expected}" ]]
}

require_count() {
  local label="$1"
  local dir="$2"
  local expected="$3"
  local count
  count=$(count_hdf5 "${dir}")
  echo "== ${label}: ${count}/${expected} hdf5 at ${dir} =="
  [[ "${count}" -ge "${expected}" ]]
}

build_core_datasets() {
  echo "== refresh libero_core_full_no_noops $(date -Is) =="
  python scripts/dataset/create_libero_core_full.py --dataset_root dataset_all/
  require_count core_full "${ROOT}/dataset_all/libero_core_full_no_noops" 10

  echo "== refresh libero_core_lt_no_noops $(date -Is) =="
  python scripts/dataset/create_libero_core_lt.py --source_dir ./dataset_all/libero_core_full_no_noops/
  require_count core_lt "${ROOT}/dataset_all/libero_core_lt_no_noops" 10
}

build_apa_hdf5() {
  if require_count core_apa "${ROOT}/dataset_all/libero_core_lt_no_noops_target_apa" 10; then
    echo "== APA hdf5 already complete =="
    return 0
  fi
  echo "== APA segmentation $(date -Is) =="
  python scripts/APA/segmentation.py --source_dir dataset_all/libero_core_lt_no_noops
  echo "== APA grafting $(date -Is) =="
  python -m scripts.APA.grafting --source_dir dataset_all/libero_core_lt_no_noops_target_approaching_phase
  echo "== APA formatting $(date -Is) =="
  python scripts/APA/formatting.py \
    --source-dir dataset_all/libero_core_lt_no_noops \
    --grafted-dir dataset_all/libero_core_lt_no_noops_target_approaching_phase_grafting \
    --target-dir dataset_all/libero_core_lt_no_noops_target_apa
  require_count core_apa "${ROOT}/dataset_all/libero_core_lt_no_noops_target_apa" 10
}

build_apa_tfds() {
  if [[ -d "${DATA_ROOT}/libero_core_apa/1.0.0" ]]; then
    echo "== skip TFDS libero_core_apa; already exists =="
    return 0
  fi
  echo "== build TFDS libero_core_apa $(date -Is) =="
  (cd rlds_dataset_builder/libero_core_apa && TFDS_DATA_DIR="${DATA_ROOT}" tfds build --data_dir "${DATA_ROOT}")
}

dry_run_apa() {
  echo "== dry-run libero_core_apa $(date -Is) =="
  python - <<'PY'
from prismatic.vla.datasets.rlds.oxe.configs import OXE_DATASET_CONFIGS
from prismatic.vla.datasets.rlds.oxe.mixtures import OXE_NAMED_MIXTURES
from prismatic.vla.datasets.rlds.oxe.transforms import OXE_STANDARDIZATION_TRANSFORMS
import tensorflow_datasets as tfds

name = "libero_core_apa"
print("config", name in OXE_DATASET_CONFIGS)
print("mixture", OXE_NAMED_MIXTURES.get(name))
print("transform", name in OXE_STANDARDIZATION_TRANSFORMS)
builder = tfds.builder_from_directory("/mnt/data/cyh/tensorflow_datasets/libero_core_apa/1.0.0")
print("splits", builder.info.splits)
PY
}

latest_step_for_run() {
  local run_dir="$1"
  python - "$run_dir/checkpoints" <<'PY'
import re, sys
from pathlib import Path
ckpt_dir = Path(sys.argv[1])
best = None
for path in ckpt_dir.glob("step-*.pt"):
    m = re.search(r"step-(\d+)-", path.name)
    if m:
        best = max(best or 0, int(m.group(1)))
if not best:
    raise SystemExit("no checkpoints found")
print(best)
PY
}

link_eval_steps() {
  local run_dir="$1"
  local step="$2"
  local ckpt_dir="${run_dir}/checkpoints"
  local ckpt_file
  ckpt_file=$(find "${ckpt_dir}" -maxdepth 1 -type f -name "step-*.pt" | sort | tail -n 1)
  local ckpt_base suffix alias dest
  ckpt_base=$(basename "${ckpt_file}")
  suffix="${ckpt_base#step-}"
  suffix="${suffix#*-}"
  for alias in "${step}" "$(printf "%05d" "${step}")" "$(printf "%06d" "${step}")"; do
    dest="step-${alias}-${suffix}"
    [[ "${dest}" == "${ckpt_base}" ]] || ln -sf "${ckpt_base}" "${ckpt_dir}/${dest}"
  done
}

train_apa_seed7() {
  local run_id="apa_libero_core_apa_seed7_b20_${RUN_STAMP}"
  if [[ -d "${RUN_ROOT}/${run_id}/checkpoints" ]] && find "${RUN_ROOT}/${run_id}/checkpoints" -maxdepth 1 -name "step-*.pt" | grep -q .; then
    echo "== skip APA train; checkpoint exists for ${run_id} =="
    return 0
  fi
  wait_for_gpus
  echo "== train APA seed7 $(date -Is) run_id=${run_id} =="
  torchrun --nnodes 1 --nproc-per-node 2 --master_addr 127.0.0.1 --master_port 31817 vla_scripts/train.py \
    --pretrained_checkpoint pretrained/minivla-libero90-prismatic/checkpoints/step-122500-epoch-55-loss=0.0743.pt \
    --vla.type "prism-qwen25-dinosiglip-224px+0_5b+mx-libero-core-lt" \
    --vla.data_mix libero_core_apa \
    --vla.expected_world_size 2 \
    --vla.global_batch_size 20 \
    --vla.per_device_batch_size 10 \
    --data_root_dir "${DATA_ROOT}" \
    --run_root_dir "${RUN_ROOT}" \
    --run_id "${run_id}" \
    --save_interval 5000 \
    --seed 7 \
    --tcad_lambda 0.0 \
    --tcad_ratio 0.0 \
    --rare_bc_max_count 0 \
    --rare_bc_weight 1.0
}

eval_apa_seed7() {
  local run_id="apa_libero_core_apa_seed7_b20_${RUN_STAMP}"
  local run_dir="${RUN_ROOT}/${run_id}"
  local step
  step=$(latest_step_for_run "${run_dir}")
  link_eval_steps "${run_dir}" "${step}"
  wait_for_gpus
  local save_root="${RESULT_ROOT}/${run_id}/${step}/apa_seed7_libero_core_30trials_${RUN_STAMP}"
  if [[ -f "${save_root}/libero_core-prismatic/step_${step}-vqa_False/000.log" ]]; then
    echo "== skip APA eval; log exists =="
    return 0
  fi
  echo "== eval APA seed7 $(date -Is) step=${step} save_root=${save_root} =="
  python vla_scripts/parallel_libero_evaluator_egl.py \
    --num-trails-per-task 30 \
    --num-gpus 2 \
    --num-processes 10 \
    --task-suite-name libero_core \
    --pretrained-checkpoint "${run_dir}" \
    --unnorm_key libero_core_apa \
    --save-root "${save_root}" \
    --steps "${step}" \
    --instruction-formatting False
  grep -R -E "Overall success rate|Task .*success rate" "${save_root}" 2>/dev/null || true
}

{
  echo "== Core-LT APA robust full-raw protocol start $(date -Is) stamp=${RUN_STAMP} =="
  cd "${ROOT}"

  for file in "${OBJECT_FILES[@]}"; do
    download_one libero_object "${file}"
  done
  for file in "${GOAL_FILES[@]}"; do
    download_one libero_goal "${file}"
  done

  echo "== raw file counts =="
  find "${ROOT}/libero_raw/libero_object" -maxdepth 1 -type f -name "*.hdf5" -printf "%f %s\n" | sort
  find "${ROOT}/libero_raw/libero_goal" -maxdepth 1 -type f -name "*.hdf5" -printf "%f %s\n" | sort

  source /opt/miniconda3/etc/profile.d/conda.sh
  conda activate "${ENV}"
  export CUDA_VISIBLE_DEVICES=2,3
  export EVAL_ALLOWED_GPUS=2,3
  export MUJOCO_GL=egl
  export PYOPENGL_PLATFORM=egl
  export HF_ENDPOINT=https://hf-mirror.com
  export HF_HOME=/mnt/data/cyh/.cache/huggingface
  export TRANSFORMERS_CACHE=/mnt/data/cyh/.cache/huggingface
  export TOKENIZERS_PARALLELISM=false
  export WANDB_DISABLED=true
  export PRISMATIC_DATA_ROOT="${ROOT}/data/prismatic"
  export PYTHONPATH="${ROOT}/LIBERO:${ROOT}:${PYTHONPATH:-}"
  export PYTHONUNBUFFERED=1
  export HF_HUB_DISABLE_TELEMETRY=1
  export TFDS_DATA_DIR="${DATA_ROOT}"
  mkdir -p "${PRISMATIC_DATA_ROOT}" "${HF_HOME}" "${RUN_ROOT}" "${RESULT_ROOT}"
  : > .hf_token

  wait_for_rarebc_done
  wait_for_old_apa
  regenerate_suite_full libero_goal 10
  regenerate_suite_full libero_object 10
  build_core_datasets
  build_apa_hdf5
  build_apa_tfds
  dry_run_apa
  train_apa_seed7
  eval_apa_seed7
  echo "== Core-LT APA robust full-raw protocol done $(date -Is) =="
} >> "${LOG}" 2>&1
