#!/usr/bin/env python3
"""Summarize Core-LT reviewer-priority evidence from LIBERO eval logs.

The script is intentionally file-system based: it can be run on server23 after
new evaluations finish, or locally against a copied result tree. It parses the
standard LIBERO `000.log` lines:
  Task i ... success rate: x
  Overall success rate: x
and emits both JSON and Markdown tables with pending cells for jobs that have
not produced a log yet.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

TASK_RE = re.compile(r"Task\s+(\d+)\s+(.+?)\s+success rate:\s+([0-9.]+)")
OVERALL_RE = re.compile(r"Overall success rate:\s+([0-9.]+)")


@dataclass(frozen=True)
class Target:
    key: str
    label: str
    group: str
    seed: int | None
    role: str
    patterns: tuple[str, ...]
    note: str = ""


TARGETS: tuple[Target, ...] = (
    Target(
        key="bc_seed7_legacy",
        label="BC seed7 legacy local",
        group="main_seed7",
        seed=7,
        role="bc",
        patterns=(
            "results/miniVLA_libero_core_lt/*x7*/17038/*30trials*/*/step_17038-vqa_False/000.log",
        ),
        note="Existing local BC evidence; not the strict two-GPU matched BC cell.",
    ),
    Target(
        key="matched_bc_seed7",
        label="BC seed7 matched two-GPU",
        group="2x2_seed7",
        seed=7,
        role="bc",
        patterns=(
            "results/core_lt_ablation/matchedbc_libero_core_lt_seed7_b20/*/*30trials*/*/step_*-vqa_False/000.log",
        ),
    ),
    Target(
        key="tcad_only_seed7",
        label="TCAD-only seed7",
        group="2x2_seed7",
        seed=7,
        role="tcad_only",
        patterns=(
            "results/rctad_lt_main/rctad_tail9_confmedian_ratio05_seed7_b20/034075/*30trials*/*/step_34075-vqa_False/000.log",
            "results/tcad_final_lt_main/tcad_final_maskpos_ratio025_seed7_b20/34075/*30trials*/*/step_34075-vqa_False/000.log",
        ),
        note="Prefer RCTAD ratio=0.5 when present; tcad_final ratio=0.25 kept as legacy diagnostic.",
    ),
    Target(
        key="rarebc_seed7",
        label="Rare-BC seed7",
        group="2x2_seed7",
        seed=7,
        role="rare_bc",
        patterns=(
            "results/core_lt_ablation/rarebc_libero_core_lt_w3_tail9_seed7_b20/*/*30trials*/*/step_*-vqa_False/000.log",
        ),
    ),
    Target(
        key="rbtad_seed7",
        label="RBTAD-TCAD seed7",
        group="main_seed7",
        seed=7,
        role="rbtad",
        patterns=(
            "results/rbtad_lt_main/rbtad_w3_tail9_confmedian_seed7_b20/034075/*30trials*/*/step_34075-vqa_False/000.log",
        ),
    ),
    Target(
        key="apa_seed7",
        label="APA seed7 same pipeline",
        group="apa",
        seed=7,
        role="apa",
        patterns=(
            "results/core_lt_priority/apa_libero_core_apa_seed7_b20/*/*30trials*/*/step_*-vqa_False/000.log",
            "results/core_lt_priority/*apa*seed7*/*/*30trials*/*/step_*-vqa_False/000.log",
        ),
    ),
    Target(
        key="apa_rbtad_seed7",
        label="APA+RBTAD seed7 same pipeline",
        group="complementarity",
        seed=7,
        role="apa_rbtad",
        patterns=(
            "results/core_lt_complementarity/apa_rbtad_libero_core_apa_w3_tail9_confmedian_seed7_b20/*/*30trials*/*/step_*-vqa_False/000.log",
        ),
    ),
    Target(
        key="bc_seed13",
        label="BC seed13",
        group="multiseed",
        seed=13,
        role="bc",
        patterns=(
            "results/core_lt_multiseed/baseline_libero_core_lt_seed13_b20/*/*30trials*/*/step_*-vqa_False/000.log",
        ),
    ),
    Target(
        key="rbtad_seed13",
        label="RBTAD-TCAD seed13",
        group="multiseed",
        seed=13,
        role="rbtad",
        patterns=(
            "results/core_lt_multiseed/rbtad_libero_core_lt_w3_tail9_confmedian_seed13_b20/*/*30trials*/*/step_*-vqa_False/000.log",
        ),
    ),
    Target(
        key="bc_seed21",
        label="BC seed21",
        group="multiseed",
        seed=21,
        role="bc",
        patterns=(
            "results/core_lt_multiseed/baseline_libero_core_lt_seed21_b20/*/*30trials*/*/step_*-vqa_False/000.log",
        ),
    ),
    Target(
        key="rbtad_seed21",
        label="RBTAD-TCAD seed21",
        group="multiseed",
        seed=21,
        role="rbtad",
        patterns=(
            "results/core_lt_multiseed/rbtad_libero_core_lt_w3_tail9_confmedian_seed21_b20/*/*30trials*/*/step_*-vqa_False/000.log",
        ),
    ),
)


def parse_log(path: Path) -> dict:
    text = path.read_text(errors="ignore")
    task_rows: dict[int, dict] = {}
    for match in TASK_RE.finditer(text):
        task_id = int(match.group(1))
        task_rows[task_id] = {
            "task_id": task_id,
            "instruction": match.group(2).strip(),
            "success": float(match.group(3)),
        }
    overall_matches = OVERALL_RE.findall(text)
    if not overall_matches:
        raise ValueError(f"missing overall success in {path}")
    tasks = [task_rows[i] for i in sorted(task_rows)]
    return {
        "path": str(path),
        "overall": float(overall_matches[-1]),
        "tasks": tasks,
        "per_task": [row["success"] for row in tasks],
        "num_tasks": len(tasks),
    }


def newest(paths: Iterable[Path]) -> Path | None:
    existing = [p for p in paths if p.exists()]
    if not existing:
        return None
    return max(existing, key=lambda p: p.stat().st_mtime)


def resolve_target(root: Path, target: Target) -> dict:
    matches: list[Path] = []
    for pattern in target.patterns:
        matches.extend(root.glob(pattern))
    chosen = newest(matches)
    result = {
        "target": asdict(target),
        "status": "pending",
        "log_path": None,
        "overall": None,
        "per_task": None,
        "num_tasks": None,
        "note": target.note,
    }
    if chosen is None:
        return result
    parsed = parse_log(chosen)
    result.update({
        "status": "done",
        "log_path": parsed["path"],
        "overall": parsed["overall"],
        "per_task": parsed["per_task"],
        "num_tasks": parsed["num_tasks"],
    })
    return result


def delta(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return round(a - b, 4)


def build_summary(root: Path) -> dict:
    entries = {target.key: resolve_target(root, target) for target in TARGETS}
    bc7 = entries["bc_seed7_legacy"]["overall"]
    matched_bc7 = entries["matched_bc_seed7"]["overall"]
    rbtad7 = entries["rbtad_seed7"]["overall"]
    seeds = []
    for seed in (7, 13, 21):
        bc_key = "matched_bc_seed7" if seed == 7 else f"bc_seed{seed}"
        if seed == 7 and entries[bc_key]["overall"] is None:
            bc_key = "bc_seed7_legacy"
        rbtad_key = f"rbtad_seed{seed}"
        if entries.get(bc_key) and entries.get(rbtad_key):
            seeds.append({
                "seed": seed,
                "bc_key": bc_key,
                "rbtad_key": rbtad_key,
                "bc": entries[bc_key]["overall"],
                "rbtad": entries[rbtad_key]["overall"],
                "delta": delta(entries[rbtad_key]["overall"], entries[bc_key]["overall"]),
                "status": "done" if entries[bc_key]["status"] == entries[rbtad_key]["status"] == "done" else "pending",
            })
    done_pairs = [s for s in seeds if s["status"] == "done"]
    avg = None
    if done_pairs:
        avg = {
            "bc": round(sum(s["bc"] for s in done_pairs) / len(done_pairs), 4),
            "rbtad": round(sum(s["rbtad"] for s in done_pairs) / len(done_pairs), 4),
            "delta": round(sum(s["delta"] for s in done_pairs) / len(done_pairs), 4),
            "n": len(done_pairs),
        }
    return {
        "root": str(root),
        "entries": entries,
        "seed7_current_gain_vs_legacy_bc": delta(rbtad7, bc7),
        "seed7_current_gain_vs_matched_bc": delta(rbtad7, matched_bc7),
        "multiseed_pairs": seeds,
        "multiseed_done_average": avg,
    }


def fmt(value: object) -> str:
    if value is None:
        return "pending"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def to_markdown(summary: dict) -> str:
    entries = summary["entries"]
    lines = ["# Core-LT Priority Evidence Summary", ""]
    lines.append("## Seed7 Main / 2x2")
    lines.append("| Cell | Overall | Per-task | Log |")
    lines.append("| --- | ---: | --- | --- |")
    for key in ("bc_seed7_legacy", "matched_bc_seed7", "tcad_only_seed7", "rarebc_seed7", "rbtad_seed7", "apa_seed7", "apa_rbtad_seed7"):
        e = entries[key]
        per_task = "pending" if e["per_task"] is None else ", ".join(f"{x:.2f}" for x in e["per_task"])
        log = e["log_path"] or "pending"
        lines.append(f"| {e['target']['label']} | {fmt(e['overall'])} | {per_task} | `{log}` |")
    lines.append("")
    lines.append(f"Seed7 RBTAD gain vs legacy BC: {fmt(summary['seed7_current_gain_vs_legacy_bc'])}")
    lines.append(f"Seed7 RBTAD gain vs matched BC: {fmt(summary['seed7_current_gain_vs_matched_bc'])}")
    lines.append("")
    lines.append("## Multiseed BC vs RBTAD")
    lines.append("| Seed | BC | RBTAD | Delta | Status |")
    lines.append("| ---: | ---: | ---: | ---: | --- |")
    for row in summary["multiseed_pairs"]:
        lines.append(f"| {row['seed']} | {fmt(row['bc'])} | {fmt(row['rbtad'])} | {fmt(row['delta'])} | {row['status']} |")
    avg = summary["multiseed_done_average"]
    if avg:
        lines.append(f"| avg n={avg['n']} | {fmt(avg['bc'])} | {fmt(avg['rbtad'])} | {fmt(avg['delta'])} | partial |")
    lines.append("")
    lines.append("## Notes")
    lines.append("- `pending` means no matching `000.log` has been produced yet.")
    lines.append("- The strict 2x2 table should prefer matched two-GPU BC when available; legacy BC is retained to protect continuity with the current draft result.")
    lines.append("- APA+RBTAD is the complementarity test: it should be compared against both same-pipeline APA and RBTAD-TCAD, not only against BC.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/mnt/data/cyh/VLA-long-tail"))
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--md-out", type=Path, default=None)
    args = parser.parse_args()

    summary = build_summary(args.root)
    json_text = json.dumps(summary, indent=2, ensure_ascii=False)
    md_text = to_markdown(summary)
    print(md_text)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json_text + "\n", encoding="utf-8")
    if args.md_out:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(md_text, encoding="utf-8")


if __name__ == "__main__":
    main()
