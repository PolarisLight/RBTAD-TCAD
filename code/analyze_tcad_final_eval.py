from pathlib import Path
import json
import re


tcad_log = Path(
    "/mnt/data/cyh/VLA-long-tail/results/tcad_final_lt_main/tcad_final_maskpos_ratio025_seed7_b20/34075/"
    "tcad_final_step34075_30trials_egl_20260608_102105/libero_core-prismatic/step_34075-vqa_False/000.log"
)
baseline_log = Path(
    "/mnt/data/cyh/VLA-long-tail/results/miniVLA_libero_core_lt/"
    "prism-qwen25-dinosiglip-224px+0_5b+mx-libero-core-lt+n0+b10+x7/17038/"
    "baseline_lt_step17038_30trials_egl_20260607_132054/libero_core-prismatic/step_17038-vqa_False/000.log"
)
tcad_debug = Path(
    "/mnt/data/cyh/VLA-long-tail/runs/tcad_final_lt_main/tcad_final_maskpos_ratio025_seed7_b20/tcad-debug.csv"
)


def parse_eval(path):
    text = path.read_text(errors="ignore")
    task_pat = re.compile(r"Task (\d+) (.*?) success rate: ([0-9.]+)")
    rows = []
    for task_id, task_name, rate in task_pat.findall(text):
        rows.append({"task_id": int(task_id), "task": task_name, "success": float(rate)})
    overall = None
    m = re.search(r"Overall success rate: ([0-9.]+)", text)
    if m:
        overall = float(m.group(1))
    return overall, rows


def parse_debug(path):
    counts = []
    losses = []
    if not path.exists():
        return {}
    for line in path.read_text().splitlines()[1:]:
        parts = line.split(",")
        if len(parts) != 4:
            continue
        counts.append(int(parts[1]))
        if parts[3] != "nan":
            losses.append(float(parts[3]))
    if not counts:
        return {}
    return {
        "num_steps_logged": len(counts),
        "mean_active_count": sum(counts) / len(counts),
        "active_step_rate": sum(c > 0 for c in counts) / len(counts),
        "mean_tcad_loss": sum(losses) / len(losses) if losses else None,
        "positive_tcad_loss_rate": sum(x > 0 for x in losses) / len(losses) if losses else None,
    }


tcad_overall, tcad_rows = parse_eval(tcad_log)
base_overall, base_rows = parse_eval(baseline_log)
base_map = {r["task_id"]: r for r in base_rows}

joined = []
for row in tcad_rows:
    base = base_map.get(row["task_id"], {})
    joined.append(
        {
            "task_id": row["task_id"],
            "task": row["task"],
            "baseline": base.get("success"),
            "tcad": row["success"],
            "delta": None if "success" not in base else row["success"] - base["success"],
        }
    )

out = {
    "baseline_overall": base_overall,
    "tcad_overall": tcad_overall,
    "overall_delta": None if base_overall is None else tcad_overall - base_overall,
    "task_rows": joined,
    "tcad_debug": parse_debug(tcad_debug),
}
print(json.dumps(out, indent=2, ensure_ascii=False))
