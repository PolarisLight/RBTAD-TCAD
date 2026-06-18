#!/usr/bin/env bash
set -euo pipefail

cd /mnt/data/cyh/VLA-long-tail
echo "== readme data/training =="
grep -nE "download|regenerate|tfds|DATA_ROOT|VLA_TYPE|train.py|libero_core|baseline|core-full|core-lt" README.md | sed -n '1,220p'

echo "== train config dataclasses =="
grep -RInE "class .*Config|data_root|dataset|vla_path|pretrained|resume|run_root|global_batch|epochs|max_steps|save" vla_scripts prismatic | head -220

echo "== builder path blocks =="
sed -n '70,140p' rlds_dataset_builder/libero_core_full/libero_core_full_dataset_builder.py
sed -n '70,140p' rlds_dataset_builder/libero_core_lt/libero_core_lt_dataset_builder.py
sed -n '70,140p' rlds_dataset_builder/libero_core_apa/libero_core_apa_dataset_builder.py
