#!/usr/bin/env bash
set -euo pipefail

cd /mnt/data/cyh/VLA-long-tail

echo "== host =="
hostname
echo "== gpu =="
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
echo "== disk =="
df -h /mnt/data
echo "== repo =="
git rev-parse --short HEAD

echo "== dataset scripts =="
find scripts/dataset rlds_dataset_builder -maxdepth 3 -type f | sort | head -200

echo "== regenerator key lines =="
grep -nE "argparse|libero-raw|target|hdf5|ProcessPool|num-gpus|demo" scripts/dataset/parallel_libero_dataset_regenerator.py | head -120

echo "== core builders key lines =="
grep -nE "source_dir|dataset_root|copy|shutil|libero_(spatial|object|goal)|glob|hdf5" scripts/dataset/create_libero_core_full.py scripts/dataset/create_libero_core_lt.py | head -200

echo "== tfds hard-coded paths =="
grep -RInE "dataset_path|/mnt|/data|/tensorflow|libero_core" rlds_dataset_builder/libero_core_full rlds_dataset_builder/libero_core_lt rlds_dataset_builder/libero_core_apa | head -200

echo "== existing data/checkpoints =="
du -sh dataset_all || true
find pretrained -maxdepth 4 -type f -printf "%p %s\n" 2>/dev/null | sort | head -50 || true
