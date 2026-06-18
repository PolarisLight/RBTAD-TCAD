from pathlib import Path
import json


run = Path("/mnt/data/cyh/VLA-long-tail/runs/tcad_final_lt_smoke/tcad_final_s5_maskpos_r2")
print(f"RUN={run}")
print(f"EXISTS={run.exists()}")

debug = run / "tcad-debug.csv"
print("== debug ==")
print(debug.read_text() if debug.exists() else "missing")

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
        for row in rows[-5:]:
            keep = {
                k: row[k]
                for k in row
                if any(token in k.lower() for token in ["step", "loss", "tcad", "lr"])
            }
            print(json.dumps(keep, ensure_ascii=False, sort_keys=True))

print("== log tail ==")
log = Path("/mnt/data/cyh/tcad_final_smoke_23.log")
if log.exists():
    for line in log.read_text(errors="ignore").splitlines()[-80:]:
        print(line)
