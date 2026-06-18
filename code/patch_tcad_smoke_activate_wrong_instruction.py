from pathlib import Path


path = Path("/mnt/data/cyh/VLA-long-tail/prismatic/vla/datasets/datasets.py")
text = path.read_text()
old = '''            # Smoke approximation of pre-contact: gripper is still open in the expert action.
            open_gripper = bool(np.asarray(action)[-1] < 0)
            if wrong_lang is not None and open_gripper:
'''
new = '''            # Smoke run: activate whenever an instruction-level target contrast exists.
            if wrong_lang is not None:
'''
if old not in text:
    raise SystemExit("tcad activation anchor not found")
text = text.replace(old, new, 1)
path.write_text(text)
print("tcad smoke activation uses wrong-instruction availability")
