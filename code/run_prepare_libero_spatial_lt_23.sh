#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/data/cyh/VLA-long-tail
ENV=/mnt/data/cyh/envs/vla-long-tail
TFDS_DATA_DIR=/mnt/data/cyh/tensorflow_datasets
RAW_DIR="$ROOT/libero_raw_hf"
LOG=/mnt/data/cyh/prepare_libero_spatial_lt_23.log

{
  echo "== prepare libero_spatial_lt start $(date -Is) =="
  source /opt/miniconda3/etc/profile.d/conda.sh
  conda activate "$ENV"
  cd "$ROOT"

  export HF_ENDPOINT=https://hf-mirror.com
  export HF_HOME=/mnt/data/cyh/.cache/huggingface
  export PYTHONPATH="$ROOT/LIBERO:$ROOT:${PYTHONPATH:-}"
  export MUJOCO_GL=egl
  export PYOPENGL_PLATFORM=egl
  export CUDA_VISIBLE_DEVICES=2
  mkdir -p "$RAW_DIR/libero_spatial" "$TFDS_DATA_DIR" "$ROOT/dataset_all"

  echo "== ensure huggingface_hub $(date -Is) =="
  python -m pip show huggingface_hub >/dev/null 2>&1 || python -m pip install "huggingface_hub[hf_xet]==0.36.2"

  echo "== list/download libero_spatial hdf5 $(date -Is) =="
  python - <<'PY' > /mnt/data/cyh/libero_spatial_hdf5_files.txt
from huggingface_hub import list_repo_files
files = list_repo_files("yifengzhu-hf/LIBERO-datasets", repo_type="dataset")
targets = sorted(f for f in files if f.startswith("libero_spatial/") and f.endswith(".hdf5"))
for f in targets:
    print(f)
PY
  echo "target_count=$(wc -l < /mnt/data/cyh/libero_spatial_hdf5_files.txt)"
  cat /mnt/data/cyh/libero_spatial_hdf5_files.txt

  while IFS= read -r relpath; do
    [ -n "$relpath" ] || continue
    out="$RAW_DIR/$relpath"
    mkdir -p "$(dirname "$out")"
    if [[ -s "$out" ]]; then
      echo "exists $out"
    else
      url="https://hf-mirror.com/datasets/yifengzhu-hf/LIBERO-datasets/resolve/main/$relpath"
      echo "download $relpath $(date -Is)"
      wget -c --tries=0 --timeout=60 --read-timeout=60 --waitretry=5 -O "$out" "$url"
    fi
    python - "$out" <<'PY'
import h5py, sys
path = sys.argv[1]
with h5py.File(path, "r") as f:
    print("validated_hdf5", path, list(f.keys())[:3])
PY
  done < /mnt/data/cyh/libero_spatial_hdf5_files.txt

  echo "== regenerate libero_spatial no_noops $(date -Is) =="
  python scripts/dataset/parallel_libero_dataset_regenerator.py \
    --num-gpus 1 \
    --max-processes 1 \
    --libero-task-suite libero_spatial \
    --libero-raw-data-dir "$RAW_DIR/libero_spatial" \
    --libero-target-dir "$ROOT/dataset_all/libero_spatial_no_noops"
  find "$ROOT/dataset_all/libero_spatial_no_noops" -maxdepth 1 -type f -name "*.hdf5" | wc -l
  du -sh "$ROOT/dataset_all/libero_spatial_no_noops"

  echo "== create libero_spatial_lt_no_noops $(date -Is) =="
  python - <<'PY'
from pathlib import Path
import h5py, random, shutil

src = Path("/mnt/data/cyh/VLA-long-tail/dataset_all/libero_spatial_no_noops")
dst = Path("/mnt/data/cyh/VLA-long-tail/dataset_all/libero_spatial_lt_no_noops")
dst.mkdir(parents=True, exist_ok=True)
counts = [46, 28, 19, 15, 11, 9, 8, 7, 6, 5]
files = sorted(src.glob("*.hdf5"))
if len(files) != 10:
    raise SystemExit(f"expected 10 spatial tasks, got {len(files)}: {[p.name for p in files]}")
random.seed(100)
for path, keep in zip(files, counts):
    out = dst / path.name
    with h5py.File(path, "r") as sf, h5py.File(out, "w") as df:
        sg = sf["data"]
        dg = df.create_group("data")
        demos = list(sg.keys())
        if keep > len(demos):
            keep = len(demos)
        keep_names = set(random.sample(demos, k=keep))
        for name in demos:
            if name in keep_names:
                sg.copy(sg[name], dg, name=name)
    print(path.name, "source", len(demos), "keep", keep)
PY
  find "$ROOT/dataset_all/libero_spatial_lt_no_noops" -maxdepth 1 -type f -name "*.hdf5" | wc -l
  du -sh "$ROOT/dataset_all/libero_spatial_lt_no_noops"

  echo "== create tfds builder libero_spatial_lt $(date -Is) =="
  rm -rf "$ROOT/rlds_dataset_builder/libero_spatial_lt"
  cp -a "$ROOT/rlds_dataset_builder/libero_core_lt" "$ROOT/rlds_dataset_builder/libero_spatial_lt"
  mv "$ROOT/rlds_dataset_builder/libero_spatial_lt/libero_core_lt_dataset_builder.py" \
     "$ROOT/rlds_dataset_builder/libero_spatial_lt/libero_spatial_lt_dataset_builder.py"
  python - <<'PY'
from pathlib import Path
root = Path("/mnt/data/cyh/VLA-long-tail/rlds_dataset_builder/libero_spatial_lt")
builder = root / "libero_spatial_lt_dataset_builder.py"
text = builder.read_text()
text = text.replace("libero_core_lt_no_noops", "libero_spatial_lt_no_noops")
text = text.replace("libero_core_lt", "libero_spatial_lt")
builder.write_text(text)
test = root / "libero_core_lt_dataset_builder_test.py"
if test.exists():
    test.rename(root / "libero_spatial_lt_dataset_builder_test.py")
    t = (root / "libero_spatial_lt_dataset_builder_test.py").read_text()
    t = t.replace("libero_core_lt", "libero_spatial_lt")
    (root / "libero_spatial_lt_dataset_builder_test.py").write_text(t)
PY

  echo "== register libero_spatial_lt in OXE configs $(date -Is) =="
  python - <<'PY'
from pathlib import Path
root = Path("/mnt/data/cyh/VLA-long-tail")
configs = root / "prismatic/vla/datasets/rlds/oxe/configs.py"
txt = configs.read_text()
if '"libero_spatial_lt"' not in txt:
    needle = '    "libero_core_lt": {\\n        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": "wrist_image"},\\n        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},\\n        "state_obs_keys": ["EEF_state", None, "gripper_state"],\\n        "state_encoding": StateEncoding.POS_EULER,\\n        "action_encoding": ActionEncoding.EEF_POS,\\n    },\\n'
    insert = needle.replace('"libero_core_lt"', '"libero_spatial_lt"')
    txt = txt.replace(needle, needle + insert)
    configs.write_text(txt)

mix = root / "prismatic/vla/datasets/rlds/oxe/mixtures.py"
txt = mix.read_text()
if '"libero_spatial_lt"' not in txt:
    needle = '    "libero_core_lt": [\\n        ("libero_core_lt", 1.0),\\n    ],\\n'
    insert = '    "libero_spatial_lt": [\\n        ("libero_spatial_lt", 1.0),\\n    ],\\n'
    txt = txt.replace(needle, needle + insert)
    mix.write_text(txt)

trans = root / "prismatic/vla/datasets/rlds/oxe/transforms.py"
txt = trans.read_text()
if '"libero_spatial_lt"' not in txt:
    needle = '    "libero_core_lt": libero_dataset_transform,\\n'
    txt = txt.replace(needle, needle + '    "libero_spatial_lt": libero_dataset_transform,\\n')
    trans.write_text(txt)
PY

  echo "== tfds build libero_spatial_lt $(date -Is) =="
  cd "$ROOT/rlds_dataset_builder/libero_spatial_lt"
  tfds build --data_dir "$TFDS_DATA_DIR"

  echo "== inspect outputs $(date -Is) =="
  find "$TFDS_DATA_DIR" -maxdepth 3 -type d | sort | grep -E "libero_spatial_lt|libero_core" || true
  du -sh "$TFDS_DATA_DIR/libero_spatial_lt" 2>/dev/null || true
  echo "== prepare libero_spatial_lt done $(date -Is) =="
} >> "$LOG" 2>&1
