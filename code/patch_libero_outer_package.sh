#!/usr/bin/env bash
set -euo pipefail

cat > /mnt/data/cyh/VLA-long-tail/LIBERO/libero/__init__.py <<'PY'
from .libero import *
from .libero import benchmark
PY

PYTHONPATH=/mnt/data/cyh/VLA-long-tail/LIBERO:/mnt/data/cyh/VLA-long-tail \
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
/mnt/data/cyh/envs/vla-long-tail/bin/python - <<'PY'
from libero.libero import benchmark as nested_benchmark
from libero import benchmark
print("nested", "libero_spatial" in nested_benchmark.get_benchmark_dict())
print("top", "libero_spatial" in benchmark.get_benchmark_dict())
PY
