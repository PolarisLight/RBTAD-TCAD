from pathlib import Path


ROOT = Path("/mnt/data/cyh/VLA-long-tail")
checks = {
    "vla_scripts/train.py": ["tcad_lambda", "TCAD_LAMBDA", "TCAD_SMOKE_STEPS"],
    "prismatic/vla/datasets/datasets.py": ["TCAD_OBJECTS", "negative_input_ids", "tcad_active"],
    "prismatic/training/strategies/base_strategy.py": [
        "def _tcad_action_logprob",
        "neg_output",
        "tcad_loss",
        "smoke_terminate",
    ],
}

for rel, needles in checks.items():
    text = (ROOT / rel).read_text()
    print(f"== {rel} ==")
    for needle in needles:
        print(f"{needle}: {'OK' if needle in text else 'MISSING'}")
