from pathlib import Path
import json

run_id = "llmlast_projector_rescue_t6_l003_s100_seed7_b20"
root = Path("/mnt/data/cyh/VLA-long-tail")
main_log = Path(f"/mnt/data/cyh/{run_id}.log")
run_dir = root / "runs" / "llmlast_projector_rescue" / run_id
result_dir = root / "results" / "llmlast_projector_rescue" / run_id

print("MAIN_LOG", main_log, main_log.exists(), main_log.stat().st_size if main_log.exists() else 0)
if main_log.exists():
    pats = [
        "TCAD",
        "tail",
        "anchor",
        "target",
        "sample_weight",
        "Trainable",
        "Loss",
        "step",
        "Overall success rate",
    ]
    lines = main_log.read_text(errors="ignore").splitlines()
    hits = []
    for i, line in enumerate(lines, 1):
        if any(p in line for p in pats):
            hits.append((i, line))
    print("LOG_HITS_HEAD")
    for i, line in hits[:160]:
        print(f"{i}: {line[:500]}")
    print("LOG_HITS_TAIL")
    for i, line in hits[-80:]:
        print(f"{i}: {line[:500]}")

print("RUN_FILES")
if run_dir.exists():
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and (
            "debug" in path.name.lower()
            or path.suffix in {".csv", ".json", ".yaml", ".yml", ".toml"}
            or path.name.startswith("config")
        ):
            rel = path.relative_to(root)
            print(f"{rel}\t{path.stat().st_size}")

debug_csv = run_dir / "tcad-debug.csv"
print("TCAD_DEBUG_SUMMARY")
if debug_csv.exists():
    import csv

    rows = list(csv.DictReader(debug_csv.open()))
    print("rows", len(rows))
    print("columns", ",".join(rows[0].keys()) if rows else "")
    print("head")
    for row in rows[:12]:
        print(json.dumps(row, ensure_ascii=False))
    print("tail")
    for row in rows[-12:]:
        print(json.dumps(row, ensure_ascii=False))
    numeric = {}
    for key in rows[0].keys() if rows else []:
        vals = []
        for row in rows:
            try:
                vals.append(float(row[key]))
            except Exception:
                pass
        if vals:
            numeric[key] = {
                "min": min(vals),
                "max": max(vals),
                "mean": sum(vals) / len(vals),
            }
    print("numeric", json.dumps(numeric, ensure_ascii=False, sort_keys=True))
else:
    print("missing")

print("CHECKPOINTS")
ckpt_dir = run_dir / "checkpoints"
if ckpt_dir.exists():
    for path in sorted(ckpt_dir.iterdir()):
        print(f"{path.name}\t{path.stat().st_size}")

print("RESULT_LOGS")
if result_dir.exists():
    logs = sorted(result_dir.rglob("000.log"))
    for path in logs[-5:]:
        print(path.relative_to(root))
        tail = path.read_text(errors="ignore").splitlines()[-20:]
        for line in tail:
            if "success rate" in line or "Overall success rate" in line:
                print(line)
