#!/usr/bin/env bash
set -euo pipefail

/mnt/data/cyh/envs/vla-long-tail/bin/python - <<'PY'
from pathlib import Path
path = Path("/mnt/data/cyh/VLA-long-tail/scripts/dataset/parallel_libero_dataset_regenerator.py")
text = path.read_text()
old = "from libero.libero import benchmark"
new = "from libero import benchmark"
if old in text:
    path.write_text(text.replace(old, new))
    print("patched benchmark import")
else:
    print("benchmark import already patched or old line missing")
for line in path.read_text().splitlines()[:16]:
    print(line)
PY
