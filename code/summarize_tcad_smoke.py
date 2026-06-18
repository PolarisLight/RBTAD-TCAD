from pathlib import Path
import json


run = Path("/mnt/data/cyh/VLA-long-tail/runs/tcad_lite_lt_smoke/tcad_lite_smoke_2gpu_s20_r1")
print(f"RUN={run}")
if not run.exists():
    raise SystemExit("run dir missing")

print("== files ==")
for path in sorted(run.rglob("*")):
    if path.is_file():
        print(f"{path.relative_to(run)} {path.stat().st_size}")

print("== metrics ==")
metrics = sorted(run.rglob("*.jsonl"))
for path in metrics:
    print(f"metrics_file={path}")
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
