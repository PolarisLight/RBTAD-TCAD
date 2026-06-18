from pathlib import Path


ROOT = Path("/mnt/data/cyh/VLA-long-tail")


def patch_train():
    path = ROOT / "vla_scripts/train.py"
    text = path.read_text()

    if "tcad_ratio: float" not in text:
        text = text.replace(
            "    tcad_lambda: float = 0.0                                        # TCAD ranking loss weight\n",
            "    tcad_lambda: float = 0.0                                        # TCAD ranking loss weight\n"
            "    tcad_ratio: float = 0.25                                        # TCAD candidate sampling ratio\n",
            1,
        )

    text = text.replace(
        '    os.environ["TCAD_RATIO"] = "1.0" if cfg.tcad_lambda > 0 else "0.0"\n',
        '    os.environ["TCAD_RATIO"] = str(cfg.tcad_ratio if cfg.tcad_lambda > 0 else 0.0)\n',
        1,
    )
    path.write_text(text)


def patch_dataset():
    path = ROOT / "prismatic/vla/datasets/datasets.py"
    text = path.read_text()

    old = '''            # Smoke run: activate whenever an instruction-level target contrast exists.
            if wrong_lang is not None:
'''
    new = '''            # Main experiment eligibility: use an action-derived gripper phase signal.
            # On LIBERO core-lt TFDS, action[-1] is binary {-1, +1}; positive starts after an initial approach phase.
            gripper_positive = bool(float(np.asarray(action)[-1]) > 0)
            if wrong_lang is not None and gripper_positive:
'''
    if old in text:
        text = text.replace(old, new, 1)
    elif "gripper_positive = bool(float(np.asarray(action)[-1]) > 0)" not in text:
        raise SystemExit("tcad activation anchor not found")

    path.write_text(text)


def main():
    patch_train()
    patch_dataset()
    print("tcad-final main patch applied")


if __name__ == "__main__":
    main()
