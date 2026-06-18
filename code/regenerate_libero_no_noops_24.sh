#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/data/cyh/VLA-long-tail
ENV=/mnt/data/cyh/envs/vla-long-tail
RAW="$ROOT/libero_raw_hf"
LOG=/mnt/data/cyh/vla_regenerate_no_noops_24.log

{
  echo "== regenerate start $(date -Is) =="
  source /opt/miniconda3/etc/profile.d/conda.sh
  conda activate "$ENV"
  cd "$ROOT"
  export MUJOCO_GL=egl
  export PYOPENGL_PLATFORM=egl
  export PYTHONPATH="$ROOT/LIBERO:$ROOT:${PYTHONPATH:-}"

  for suite in libero_spatial libero_object libero_goal; do
    echo "== suite $suite $(date -Is) =="
    python scripts/dataset/parallel_libero_dataset_regenerator.py \
      --num-gpus 1 \
      --max-processes 1 \
      --libero-task-suite "$suite" \
      --libero-raw-data-dir "$RAW/$suite" \
      --libero-target-dir "$ROOT/dataset_all/${suite}_no_noops"
    echo "== suite $suite done $(date -Is) =="
    find "$ROOT/dataset_all/${suite}_no_noops" -maxdepth 1 -type f -name "*.hdf5" | wc -l
    du -sh "$ROOT/dataset_all/${suite}_no_noops"
  done

  echo "== regenerate done $(date -Is) =="
} >> "$LOG" 2>&1
