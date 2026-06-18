#!/usr/bin/env bash
set -euo pipefail

echo "== processes =="
pgrep -af 'download_libero_hdf5_sequential|wget -c|tfds build|parallel_libero_dataset_regenerator|vla_scripts/train.py' || true

echo "== gpu =="
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader

echo "== download log tail =="
tail -80 /mnt/data/cyh/vla_libero_hdf5_sequential.log 2>/dev/null || true

echo "== raw sizes/counts =="
du -sh /mnt/data/cyh/VLA-long-tail/libero_raw_hf 2>/dev/null || true
for d in libero_spatial libero_object libero_goal; do
  dir="/mnt/data/cyh/VLA-long-tail/libero_raw_hf/$d"
  printf "%s " "$d"
  if [ -d "$dir" ]; then
    find "$dir" -maxdepth 1 -type f -name '*.hdf5' | wc -l
  else
    echo 0
  fi
done

echo "== validated hdf5 count =="
/mnt/data/cyh/envs/vla-long-tail/bin/python - <<'PY'
from pathlib import Path
import h5py
root = Path("/mnt/data/cyh/VLA-long-tail/libero_raw_hf")
ok = []
bad = []
for p in sorted(root.glob("*/*.hdf5")):
    try:
        with h5py.File(p, "r") as f:
            _ = list(f.keys())[:1]
        ok.append(p)
    except Exception as exc:
        bad.append((p, str(exc)))
print("ok", len(ok))
print("bad", len(bad))
for p, exc in bad[:5]:
    print("bad_file", p, exc)
PY
