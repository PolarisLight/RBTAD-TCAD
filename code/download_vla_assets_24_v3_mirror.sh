#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/data/cyh/VLA-long-tail
ENV=/mnt/data/cyh/envs/vla-long-tail
LOG=/mnt/data/cyh/vla_assets_download_24_mirror.log
RAW_DIR="$ROOT/libero_raw_hf"
CKPT_DIR="$ROOT/pretrained/minivla-libero90-prismatic"

{
  echo "== mirror start $(date -Is) =="
  source /opt/miniconda3/etc/profile.d/conda.sh
  conda activate "$ENV"

  python -m pip install "huggingface_hub[hf_xet]==0.36.2"
  export HF_ENDPOINT=https://hf-mirror.com

  mkdir -p "$CKPT_DIR" "$RAW_DIR"

  echo "== list target dataset files $(date -Is) =="
  python - <<'PY'
from huggingface_hub import list_repo_files
files = list_repo_files("yifengzhu-hf/LIBERO-datasets", repo_type="dataset")
for prefix in ("libero_spatial/", "libero_object/", "libero_goal/"):
    subset = [f for f in files if f.startswith(prefix) and f.endswith(".hdf5")]
    print(prefix, len(subset))
    print("\n".join(subset[:3]))
PY

  echo "== checkpoint mirror download $(date -Is) =="
  huggingface-cli download Stanford-ILIAD/minivla-libero90-prismatic \
    checkpoints/step-122500-epoch-55-loss=0.0743.pt \
    --local-dir "$CKPT_DIR" \
    --resume-download

  echo "== libero hdf5 mirror download $(date -Is) =="
  huggingface-cli download yifengzhu-hf/LIBERO-datasets \
    --repo-type dataset \
    --include "libero_spatial/*.hdf5" \
    --include "libero_object/*.hdf5" \
    --include "libero_goal/*.hdf5" \
    --local-dir "$RAW_DIR" \
    --resume-download

  echo "== downloaded sizes $(date -Is) =="
  du -sh "$CKPT_DIR" "$RAW_DIR"
  find "$CKPT_DIR" -maxdepth 3 -type f -printf "%p %s\n" | sort | tail -20
  find "$RAW_DIR" -maxdepth 2 -type f -name "*.hdf5" | wc -l
  find "$RAW_DIR" -maxdepth 2 -type f -name "*.hdf5" -printf "%p %s\n" | sort | head -20
  echo "== mirror done $(date -Is) =="
} >> "$LOG" 2>&1
