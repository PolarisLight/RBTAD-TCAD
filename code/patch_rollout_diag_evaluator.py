from pathlib import Path

PATHS = [Path("vla_scripts/parallel_libero_evaluator_egl.py")]

for path in PATHS:
    text = path.read_text(encoding="utf-8")
    if 'os.environ.setdefault("CUDA_VISIBLE_DEVICES"' not in text:
        text = text.replace('os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  ', 'os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3")', 1)
    if "import json" not in text.splitlines()[:20]:
        text = text.replace("import argparse\n", "import argparse\nimport json\n", 1)
    if "def _to_diag_list" not in text:
        helper = '''

def _to_diag_list(value):
    if value is None:
        return None
    try:
        arr = np.asarray(value)
        return arr.astype(float).reshape(-1).tolist()
    except Exception:
        return str(value)


def _diag_obs_state(obs):
    return {
        "eef_pos": _to_diag_list(obs.get("robot0_eef_pos")),
        "eef_quat": _to_diag_list(obs.get("robot0_eef_quat")),
        "gripper_qpos": _to_diag_list(obs.get("robot0_gripper_qpos")),
    }
'''
        text = text.replace("\n\ndef get_image_resize_size(cfg):\n", helper + "\n\ndef get_image_resize_size(cfg):\n", 1)
    if "diag_file = os.environ.get(\"ROLLOUT_DIAG_FILE\")" not in text:
        text = text.replace(
'''        replay_images, replay_wrist_images = [], []
        texts = []
        timestep = 0
        success = False
''',
'''        replay_images, replay_wrist_images = [], []
        texts = []
        diag_file = os.environ.get("ROLLOUT_DIAG_FILE")
        diag_dir = os.environ.get("ROLLOUT_DIAG_DIR")
        diag_steps = []
        close_step = None
        action_norm_sum = 0.0
        action_count = 0
        timestep = 0
        success = False
''', 1)
        text = text.replace(
'''                obs, reward, done, info = env.step(action.tolist())
                self._add_observation(obs, replay_images, replay_wrist_images)

                timestep += 1
''',
'''                action_arr = np.asarray(action, dtype=float).reshape(-1)
                obs, reward, done, info = env.step(action.tolist())
                self._add_observation(obs, replay_images, replay_wrist_images)
                action_count += 1
                action_norm_sum += float(np.linalg.norm(action_arr[:6])) if action_arr.size >= 6 else float(np.linalg.norm(action_arr))
                if close_step is None and action_arr.size > 0 and action_arr[-1] > 0:
                    close_step = timestep
                if diag_file and (timestep % 10 == 0 or done):
                    diag_steps.append({"timestep": int(timestep), "action": _to_diag_list(action_arr), "done": bool(done), "reward": float(reward), "state": _diag_obs_state(obs)})

                timestep += 1
''', 1)
        text = text.replace(
'''        # Skip GIF writing on headless servers without ffmpeg; metrics only need success summaries.
        self.logger.info(f'Task {task_id} {task_description} episode {episode}: success {success}')
        return {"task_id": task_id, "task": task_description, "episode": episode, "success": success}
''',
'''        if diag_file:
            record = {
                "task_id": int(task_id),
                "task": task_description,
                "episode": int(episode),
                "init_id": int(init_id),
                "success": bool(success),
                "steps": int(timestep),
                "close_step": None if close_step is None else int(close_step),
                "action_count": int(action_count),
                "mean_action_norm": float(action_norm_sum / max(action_count, 1)),
                "final_state": _diag_obs_state(obs),
                "sampled_steps": diag_steps[-8:],
            }
            if diag_dir and (not success) and replay_images:
                os.makedirs(diag_dir, exist_ok=True)
                image_path = os.path.join(diag_dir, f"task{task_id:02d}_ep{episode:03d}_final.png")
                Image.fromarray(replay_images[-1]).save(image_path)
                record["final_image"] = image_path
            with open(diag_file, "a") as f:
                f.write(json.dumps(record) + "\\n")
        # Skip GIF writing on headless servers without ffmpeg; metrics only need success summaries.
        self.logger.info(f'Task {task_id} {task_description} episode {episode}: success {success}')
        return {"task_id": task_id, "task": task_description, "episode": episode, "success": success}
''', 1)
    if 'EVAL_ALLOWED_GPUS' not in text:
        text = text.replace(
'''        used_memorys = os.popen(f"nvidia-smi --query-gpu=memory.used --format=csv,nounits,noheader").readlines()
        used_memorys = [int(memory.strip()) for memory in used_memorys]
        return [i for i, memory in enumerate(used_memorys) if memory < 1000]
''',
'''        used_memorys = os.popen(f"nvidia-smi --query-gpu=memory.used --format=csv,nounits,noheader").readlines()
        used_memorys = [int(memory.strip()) for memory in used_memorys]
        allowed_env = os.environ.get("EVAL_ALLOWED_GPUS", "").strip()
        allowed = None
        if allowed_env:
            allowed = {int(item) for item in allowed_env.split(",") if item.strip().isdigit()}
        return [i for i, memory in enumerate(used_memorys) if memory < 1000 and (allowed is None or i in allowed)]
''', 1)
    path.write_text(text, encoding="utf-8")
    print(f"patched {path}")
