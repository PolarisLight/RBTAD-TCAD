#!/usr/bin/env bash
set -euo pipefail

/mnt/data/cyh/envs/vla-long-tail/bin/python - <<'PY'
from pathlib import Path
path = Path("/mnt/data/cyh/VLA-long-tail/scripts/dataset/parallel_libero_dataset_regenerator.py")
text = path.read_text()
old = 'os.environ["MUJOCO_GL"] = "osmesa"'
new = 'os.environ.setdefault("MUJOCO_GL", "egl")'
if old in text:
    path.write_text(text.replace(old, new))
    print("patched regenerator to default MUJOCO_GL=egl")
else:
    print("regenerator already patched or old line missing")
print(path.read_text().splitlines()[:4])
PY
