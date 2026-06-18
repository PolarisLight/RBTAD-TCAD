#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/data/cyh/VLA-long-tail
ENV=/mnt/data/cyh/envs/vla-long-tail
LOG=/mnt/data/cyh/vla_assets_download_24.log
RAW_DIR="$ROOT/libero_raw_zips"
CKPT_DIR="$ROOT/pretrained/minivla-libero90-prismatic"

{
  echo "== start $(date -Is) =="
  echo "host=$(hostname)"
  df -h /mnt/data

  source /opt/miniconda3/etc/profile.d/conda.sh
  conda activate "$ENV"
  python -m pip install -U "huggingface_hub[hf_xet]"

  mkdir -p "$CKPT_DIR" "$RAW_DIR"

  echo "== checkpoint download $(date -Is) =="
  huggingface-cli download Stanford-ILIAD/minivla-libero90-prismatic \
    checkpoints/step-122500-epoch-55-loss=0.0743.pt \
    --local-dir "$CKPT_DIR" \
    --resume-download

  echo "== libero zip download $(date -Is) =="
  cd "$RAW_DIR"
  wget -c -O libero_spatial.zip "https://utexas.box.com/shared/static/04k94hyizn4huhbv5sz4ev9p2h1p6s7f.zip"
  wget -c -O libero_object.zip "https://utexas.box.com/shared/static/avkklgeq0e1dgzxz52x488whpu8mgspk.zip"
  wget -c -O libero_goal.zip "https://utexas.box.com/shared/static/iv5e4dos8yy2b212pkzkpxu9wbdgjfeg.zip"

  echo "== sizes $(date -Is) =="
  du -sh "$CKPT_DIR" "$RAW_DIR"
  find "$CKPT_DIR" -maxdepth 3 -type f -printf "%p %s\n" | sort
  find "$RAW_DIR" -maxdepth 1 -type f -printf "%p %s\n" | sort
  echo "== done $(date -Is) =="
} >> "$LOG" 2>&1
