from pathlib import Path

path = Path("vla_scripts/parallel_libero_evaluator_egl.py")
text = path.read_text(encoding="utf-8")

if "def _apply_tail_action_gain" not in text:
    helper = '''

def _apply_tail_action_gain(action, task_id):
    gain = float(os.environ.get("TAIL_ACTION_GAIN", "1.0") or "1.0")
    if abs(gain - 1.0) < 1e-8:
        return action
    ids_env = os.environ.get("TAIL_ACTION_TASK_IDS", "").strip()
    if ids_env:
        ids = {int(item) for item in ids_env.split(",") if item.strip().lstrip("-").isdigit()}
        if int(task_id) not in ids:
            return action

    def _scale_one(value):
        arr = np.asarray(value, dtype=float).copy()
        if arr.size >= 6:
            arr[..., :6] = np.clip(arr[..., :6] * gain, -1.0, 1.0)
        return arr

    if isinstance(action, list):
        return [_scale_one(item) for item in action]
    return _scale_one(action)
'''
    text = text.replace("\n\ndef invert_gripper_action(action):\n", helper + "\n\ndef invert_gripper_action(action):\n", 1)

needle = '''            if isinstance(action, list):
                for a in action:
                    obs, reward, done, info = env.step(a.tolist())
'''
replacement = '''            action = _apply_tail_action_gain(action, task_id)

            if isinstance(action, list):
                for a in action:
                    obs, reward, done, info = env.step(a.tolist())
'''
if replacement not in text:
    if needle not in text:
        raise SystemExit("action execution insertion point not found")
    text = text.replace(needle, replacement, 1)

path.write_text(text, encoding="utf-8")
print(f"patched {path} with tail action gain support")