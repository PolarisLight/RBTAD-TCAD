#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/data/cyh/VLA-long-tail
ENV=/mnt/data/cyh/envs/vla-long-tail
TFDS_DATA_DIR=/mnt/data/cyh/tensorflow_datasets
LOG=/mnt/data/cyh/vla_build_tfds_only_24.log

{
  echo "== tfds only start $(date -Is) =="
  source /opt/miniconda3/etc/profile.d/conda.sh
  conda activate "$ENV"
  export PYTHONPATH="$ROOT/LIBERO:$ROOT:${PYTHONPATH:-}"
  mkdir -p "$TFDS_DATA_DIR"

  echo "== tfds version =="
  python - <<'PY'
import tensorflow_datasets as tfds
print(tfds.__version__)
PY

  echo "== build full $(date -Is) =="
  cd "$ROOT/rlds_dataset_builder/libero_core_full"
  tfds build --data_dir "$TFDS_DATA_DIR"

  echo "== build lt $(date -Is) =="
  cd "$ROOT/rlds_dataset_builder/libero_core_lt"
  tfds build --data_dir "$TFDS_DATA_DIR"

  echo "== inspect tfds $(date -Is) =="
  find "$TFDS_DATA_DIR" -maxdepth 3 -type d | sort | grep -E "libero_core_(full|lt)" || true
  find "$TFDS_DATA_DIR" -maxdepth 4 -type f | grep -E "libero_core_(full|lt)" | head -80 || true
  du -sh "$TFDS_DATA_DIR"
  echo "== tfds only done $(date -Is) =="
} >> "$LOG" 2>&1
