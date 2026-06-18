from pathlib import Path
import re

root = Path("/mnt/data/cyh/VLA-long-tail")
patterns = [
    "libero_core_lt",
    "core_lt",
    "libero_spatial",
    "spatial",
    "libero_object",
    "object",
    "libero_goal",
    "goal",
    "data_mix",
    "unnorm_key",
    "task-suite-name",
]
dirs = ["prismatic", "vla_scripts", "scripts", "configs", "LIBERO"]

for d in dirs:
    base = root / d
    if not base.exists():
        continue
    for path in base.rglob("*"):
        if not path.is_file() or path.suffix not in {".py", ".yaml", ".yml", ".json", ".sh", ".md", ".txt"}:
            continue
        try:
            text = path.read_text(errors="ignore")
        except Exception:
            continue
        hits = []
        for i, line in enumerate(text.splitlines(), 1):
            low = line.lower()
            if any(p in low for p in patterns):
                hits.append((i, line.strip()))
        if hits:
            print(f"\n## {path.relative_to(root)}")
            for i, line in hits[:80]:
                print(f"{i}: {line[:240]}")
