from pathlib import Path

logs = {
    "baseline": Path("/mnt/data/cyh/VLA-long-tail/results/cross_dataset_probe/baseline_libero_core_s1000_seed7_b20/1000/baseline_libero_core_10trials_egl_20260612_190846/libero_core-prismatic/step_1000-vqa_False/000.log"),
    "rbtad": Path("/mnt/data/cyh/VLA-long-tail/results/cross_dataset_probe/rbtad_libero_core_s1000_seed7_b20/1000/rbtad_libero_core_10trials_egl_20260612_202903/libero_core-prismatic/step_1000-vqa_False/000.log"),
}

for name, path in logs.items():
    print(f"## {name} {path}")
    lines = path.read_text(errors="ignore").splitlines()
    for line in lines:
        if "success rate:" in line or "Overall success rate:" in line:
            print(line)

print("## checkpoints")
for rel in [
    "runs/cross_dataset_probe/baseline_libero_core_s1000_seed7_b20/checkpoints",
    "runs/cross_dataset_probe/rbtad_libero_core_s1000_seed7_b20/checkpoints",
]:
    path = Path("/mnt/data/cyh/VLA-long-tail") / rel
    print(rel)
    for child in sorted(path.iterdir()):
        print(child.name, child.stat().st_size)
