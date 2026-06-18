#!/usr/bin/env bash
set -euo pipefail
cd /mnt/data/cyh/VLA-long-tail
sed -n '115,230p' vla_scripts/train.py
sed -n '680,715p' prismatic/vla/datasets/rlds/oxe/configs.py
sed -n '1,40p' prismatic/vla/datasets/rlds/oxe/mixtures.py
