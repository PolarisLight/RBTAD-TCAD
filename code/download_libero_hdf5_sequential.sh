#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/data/cyh/VLA-long-tail
ENV=/mnt/data/cyh/envs/vla-long-tail
RAW_DIR="$ROOT/libero_raw_hf"
LOG=/mnt/data/cyh/vla_libero_hdf5_sequential.log
LIST=/mnt/data/cyh/libero_core_hdf5_files.txt

{
  echo "== sequential start $(date -Is) =="
  source /opt/miniconda3/etc/profile.d/conda.sh
  conda activate "$ENV"
  export HF_ENDPOINT=https://hf-mirror.com
  mkdir -p "$RAW_DIR"

  python - <<'PY' > /mnt/data/cyh/libero_core_hdf5_files.txt
from huggingface_hub import list_repo_files
files = list_repo_files("yifengzhu-hf/LIBERO-datasets", repo_type="dataset")
targets = []
for prefix in ("libero_spatial/", "libero_object/", "libero_goal/"):
    targets.extend(sorted(f for f in files if f.startswith(prefix) and f.endswith(".hdf5")))
for f in targets:
    print(f)
PY

  echo "target_count=$(wc -l < "$LIST")"
  cat "$LIST"

  while IFS= read -r relpath; do
    [ -n "$relpath" ] || continue
    out="$RAW_DIR/$relpath"
    mkdir -p "$(dirname "$out")"
    url="https://hf-mirror.com/datasets/yifengzhu-hf/LIBERO-datasets/resolve/main/$relpath"
    echo "== download $relpath $(date -Is) =="
    wget -c --tries=0 --timeout=60 --read-timeout=60 --waitretry=5 -O "$out" "$url"
    python - "$out" <<'PY'
import h5py, sys
path = sys.argv[1]
with h5py.File(path, "r") as f:
    keys = list(f.keys())
    print("validated_hdf5", path, keys[:3])
PY
  done < "$LIST"

  echo "== final $(date -Is) =="
  find "$RAW_DIR" -maxdepth 2 -type f -name "*.hdf5" | wc -l
  du -sh "$RAW_DIR"
  find "$RAW_DIR" -maxdepth 2 -type f -name "*.hdf5" -printf "%p %s\n" | sort
} >> "$LOG" 2>&1
