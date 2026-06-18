from pathlib import Path

code = Path("/mnt/data/cyh/VLA-long-tail/vla_scripts/parallel_libero_evaluator_egl.py")
lines = code.read_text().splitlines()
print("CODE")
for i in range(175, 196):
    print(f"{i + 1}: {lines[i]}")

root = Path("/mnt/data/cyh/VLA-long-tail/runs/anchor_rbtad/anchor_rbtad_l2all_w125_a005_tail9_s500_seed7_b20/checkpoints")
print("FILES")
for item in sorted(root.iterdir()):
    print(item.name, "->", item.resolve() if item.is_symlink() else "")
