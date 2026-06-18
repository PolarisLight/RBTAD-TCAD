from pathlib import Path

path = Path("/mnt/data/cyh/VLA-long-tail/vla_scripts/parallel_libero_evaluator_egl.py")
lines = path.read_text().splitlines()
for start, end in [(205, 250), (250, 380)]:
    print(f"BLOCK {start + 1}-{end}")
    for idx in range(start, min(end, len(lines))):
        print(f"{idx + 1}: {lines[idx]}")
