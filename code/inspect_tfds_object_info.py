import json

import tensorflow_datasets as tfds


builder = tfds.builder_from_directory("/mnt/data/cyh/tensorflow_datasets/libero_core_lt/1.0.0")
dataset = builder.as_dataset(split="train")
for episode_idx, episode in enumerate(dataset.take(2)):
    print("EPISODE", episode_idx)
    steps = episode["steps"]
    for step_idx, step in enumerate(steps.take(5)):
        lang = step["language_instruction"].numpy().decode("utf-8")
        obj = step["object_info"].numpy().decode("utf-8", errors="replace")
        print("step", step_idx, "lang", lang)
        print("object_info", obj[:1000])
        try:
            parsed = json.loads(obj)
            print("json keys", parsed.keys() if isinstance(parsed, dict) else type(parsed))
        except Exception as exc:
            print("json parse error", type(exc).__name__, str(exc)[:120])
        action = step["action"].numpy()
        state = step["observation"]["state"].numpy()
        print("action", action, "state", state)
