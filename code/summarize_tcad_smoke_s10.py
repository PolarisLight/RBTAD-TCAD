from pathlib import Path
import json
import subprocess


run = Path("/mnt/data/cyh/VLA-long-tail/runs/tcad_lite_lt_smoke/tcad_lite_smoke_2gpu_s10_r2")
print(f"RUN={run}")
print(f"EXISTS={run.exists()}")

debug = run / "tcad-debug.csv"
print("== debug ==")
if debug.exists():
    print(debug.read_text())
else:
    print("missing")

print("== files ==")
if run.exists():
    for path in sorted(run.rglob("*")):
        if path.is_file():
            print(f"{path.relative_to(run)} {path.stat().st_size}")

print("== metrics tail ==")
if run.exists():
    for path in sorted(run.rglob("*.jsonl")):
        print(f"metrics_file={path.name}")
        rows = []
        for line in path.read_text(errors="ignore").splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        print(f"rows={len(rows)}")
        for row in rows[-3:]:
            keep = {
                k: row[k]
                for k in row
                if any(token in k.lower() for token in ["step", "loss", "tcad", "lr"])
            }
            print(json.dumps(keep, ensure_ascii=False, sort_keys=True))

print("== latest log lines ==")
log = Path("/mnt/data/cyh/tcad_lite_smoke_23.log")
if log.exists():
    lines = log.read_text(errors="ignore").splitlines()
    for line in lines[-60:]:
        print(line)
