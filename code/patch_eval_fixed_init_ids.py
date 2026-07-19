from pathlib import Path

path = Path("vla_scripts/parallel_libero_evaluator_egl.py")
text = path.read_text(encoding="utf-8")

needle = '''        init_ids = np.random.choice(50, size=self.cfg.num_trials_per_task, replace=False)
'''
replacement = '''        fixed_init_ids = os.environ.get("EVAL_INIT_IDS", "").strip()
        if fixed_init_ids:
            init_values = [int(item) for item in fixed_init_ids.split(",") if item.strip()]
            if len(init_values) < self.cfg.num_trials_per_task:
                raise ValueError(
                    f"EVAL_INIT_IDS provides {len(init_values)} ids but num_trials_per_task={self.cfg.num_trials_per_task}"
                )
            init_ids = np.array(init_values[: self.cfg.num_trials_per_task], dtype=int)
        else:
            init_ids = np.random.choice(50, size=self.cfg.num_trials_per_task, replace=False)
        self.logger.info(f"Init ids: {init_ids.tolist()}")
'''
if replacement not in text:
    if needle not in text:
        raise SystemExit("init_ids insertion point not found")
    text = text.replace(needle, replacement, 1)

path.write_text(text, encoding="utf-8")
print(f"patched {path} with fixed init-id support")