#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/data/cyh/VLA-long-tail
cd "$ROOT"

/mnt/data/cyh/envs/vla-long-tail/bin/python - <<'PY'
from pathlib import Path

root = Path("/mnt/data/cyh/VLA-long-tail")

mixtures = root / "prismatic/vla/datasets/rlds/oxe/mixtures.py"
text = mixtures.read_text()
needle = '    "libero_union4": [\n        ("libero_spatial", 1.0),\n        ("libero_object", 1.0),\n        ("libero_goal", 1.0),\n        ("libero_10", 1.0),\n    ],\n'
insert = needle + '    "libero_core_full": [\n        ("libero_core_full", 1.0),\n    ],\n    "libero_core_lt": [\n        ("libero_core_lt", 1.0),\n    ],\n    "libero_core_apa": [\n        ("libero_core_apa", 1.0),\n    ],\n'
if '"libero_core_full": [' not in text:
    if needle not in text:
        raise SystemExit("mixtures insertion point not found")
    mixtures.write_text(text.replace(needle, insert))
    print("patched mixtures.py")
else:
    print("mixtures.py already patched")

paths = {
    root / "rlds_dataset_builder/libero_core_full/libero_core_full_dataset_builder.py":
        '        path = "/mnt/data/cyh/VLA-long-tail/dataset_all/libero_core_full_no_noops"',
    root / "rlds_dataset_builder/libero_core_lt/libero_core_lt_dataset_builder.py":
        '        path = "/mnt/data/cyh/VLA-long-tail/dataset_all/libero_core_lt_no_noops"',
    root / "rlds_dataset_builder/libero_core_apa/libero_core_apa_dataset_builder.py":
        '        path = "/mnt/data/cyh/VLA-long-tail/dataset_all/libero_core_lt_no_noops_target_apa"',
}
for path, replacement in paths.items():
    text = path.read_text()
    lines = text.splitlines()
    changed = False
    out = []
    for line in lines:
        if line.strip().startswith('path = "/path/to/'):
            out.append(replacement)
            changed = True
        else:
            out.append(line)
    if changed:
        path.write_text("\n".join(out) + "\n")
        print(f"patched {path.relative_to(root)}")
    else:
        print(f"path already patched or placeholder missing: {path.relative_to(root)}")
PY

grep -n "libero_core" prismatic/vla/datasets/rlds/oxe/mixtures.py
grep -RIn 'path = "/mnt/data/cyh/VLA-long-tail/dataset_all' rlds_dataset_builder/libero_core_full rlds_dataset_builder/libero_core_lt rlds_dataset_builder/libero_core_apa
