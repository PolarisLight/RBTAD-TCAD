#!/usr/bin/env bash
set -euo pipefail

echo "== HF/Qwen cache probe $(date -Is) =="
for d in \
  /mnt/data/cyh/.cache/huggingface \
  /home/cyh/.cache/huggingface \
  /mnt/data/cyh/VLA-long-tail/pretrained \
  /mnt/data/cyh/VLA-long-tail; do
  echo "-- ${d}"
  if [[ -d "${d}" ]]; then
    find "${d}" -maxdepth 7 \( -path '*Qwen*0.5B*' -o -name 'config.json' \) -print 2>/dev/null | head -120
  else
    echo "missing"
  fi
done

echo "== selected env =="
env | grep -E '^(HF|TRANSFORMERS|CUDA|MUJOCO|TFDS)_' | sort || true
