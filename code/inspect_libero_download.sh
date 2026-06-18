#!/usr/bin/env bash
set -euo pipefail

cd /mnt/data/cyh/VLA-long-tail/LIBERO
echo "== download script =="
sed -n '1,220p' benchmark_scripts/download_libero_datasets.py
echo "== download utils =="
sed -n '1,260p' libero/libero/utils/download_utils.py
