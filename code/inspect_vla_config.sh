#!/usr/bin/env bash
set -euo pipefail

cd /mnt/data/cyh/VLA-long-tail
sed -n '150,205p' prismatic/conf/vla.py
echo "== finetune scripts =="
for f in vla_scripts/finetune/*.sh; do
  echo "--- $f"
  cat "$f"
done
echo "== train config start =="
sed -n '55,115p' vla_scripts/train.py
echo "== dataset materializer =="
grep -RInE "libero-core-full|libero-core-lt|libero_core_full|libero_core_lt|OXE|dataset_mix" prismatic/vla prismatic/conf | head -160 || true
