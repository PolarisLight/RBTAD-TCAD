#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/data/cyh/VLA-long-tail
ENV=/mnt/data/cyh/envs/vla-long-tail
TFDS_DATA_DIR=/mnt/data/cyh/tensorflow_datasets
LOG=/mnt/data/cyh/vla_build_core_tfds_24.log

{
  echo "== core/tfds start $(date -Is) =="
  source /opt/miniconda3/etc/profile.d/conda.sh
  conda activate "$ENV"
  cd "$ROOT"
  export PYTHONPATH="$ROOT/LIBERO:$ROOT:${PYTHONPATH:-}"
  mkdir -p "$TFDS_DATA_DIR"

  echo "== verify no_noops inputs =="
  for d in libero_spatial_no_noops libero_object_no_noops libero_goal_no_noops; do
    printf "%s " "$d"
    find "$ROOT/dataset_all/$d" -maxdepth 1 -type f -name "*.hdf5" | wc -l
    du -sh "$ROOT/dataset_all/$d"
  done

  echo "== create core full $(date -Is) =="
  python scripts/dataset/create_libero_core_full.py \
    --dataset_root "$ROOT/dataset_all" \
    --target_dir_name libero_core_full_no_noops
  find "$ROOT/dataset_all/libero_core_full_no_noops" -maxdepth 1 -type f -name "*.hdf5" | wc -l
  du -sh "$ROOT/dataset_all/libero_core_full_no_noops"

  echo "== create core lt $(date -Is) =="
  python scripts/dataset/create_libero_core_lt.py \
    --source_dir "$ROOT/dataset_all/libero_core_full_no_noops"
  find "$ROOT/dataset_all/libero_core_lt_no_noops" -maxdepth 1 -type f -name "*.hdf5" | wc -l
  du -sh "$ROOT/dataset_all/libero_core_lt_no_noops"

  echo "== install rlds builder editable $(date -Is) =="
  python -m pip install -e "$ROOT/rlds_dataset_builder"

  echo "== tfds build full $(date -Is) =="
  cd "$ROOT/rlds_dataset_builder/libero_core_full"
  tfds build --data_dir "$TFDS_DATA_DIR"

  echo "== tfds build lt $(date -Is) =="
  cd "$ROOT/rlds_dataset_builder/libero_core_lt"
  tfds build --data_dir "$TFDS_DATA_DIR"

  echo "== tfds outputs $(date -Is) =="
  find "$TFDS_DATA_DIR" -maxdepth 3 -type d | sort | grep -E "libero_core_(full|lt)" || true
  du -sh "$TFDS_DATA_DIR" || true
  echo "== core/tfds done $(date -Is) =="
} >> "$LOG" 2>&1
