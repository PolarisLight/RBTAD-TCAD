#!/usr/bin/env bash
set -euo pipefail
cd /mnt/data/cyh/VLA-long-tail
sed -n '1,340p' scripts/dataset/parallel_libero_dataset_regenerator.py
