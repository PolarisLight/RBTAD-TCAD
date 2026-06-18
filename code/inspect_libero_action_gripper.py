import json
from pathlib import Path

import numpy as np
import tensorflow_datasets as tfds


TFDS_DIR = "/mnt/data/cyh/tensorflow_datasets/libero_core_lt/1.0.0"


def main():
    builder = tfds.builder_from_directory(TFDS_DIR)
    dataset = builder.as_dataset(split="train")

    values = []
    per_episode = []
    signs = {"lt0": 0, "eq0": 0, "gt0": 0}
    examples = []

    for episode_idx, episode in enumerate(dataset.take(200)):
        steps = list(episode["steps"].as_numpy_iterator())
        actions = np.stack([step["action"] for step in steps])
        last = actions[:, -1].astype(float)
        values.extend(last.tolist())
        signs["lt0"] += int((last < 0).sum())
        signs["eq0"] += int((last == 0).sum())
        signs["gt0"] += int((last > 0).sum())

        unique = sorted(set(np.round(last, 4).tolist()))
        per_episode.append(
            {
                "episode_idx": episode_idx,
                "n": int(len(last)),
                "min": float(last.min()),
                "max": float(last.max()),
                "mean": float(last.mean()),
                "unique_head": unique[:8],
                "unique_tail": unique[-8:],
                "first_gt0": int(np.where(last > 0)[0][0]) if np.any(last > 0) else None,
                "first_lt0": int(np.where(last < 0)[0][0]) if np.any(last < 0) else None,
            }
        )
        if len(examples) < 12:
            instr = steps[0]["language_instruction"].decode("utf-8")
            examples.append({"instruction": instr, "last_values_head": last[:20].tolist()})

    arr = np.asarray(values, dtype=float)
    out = {
        "num_values": int(arr.size),
        "signs": signs,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "percentiles": {str(p): float(np.percentile(arr, p)) for p in [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]},
        "episodes_with_gt0": int(sum(row["first_gt0"] is not None for row in per_episode)),
        "episodes_with_lt0": int(sum(row["first_lt0"] is not None for row in per_episode)),
        "per_episode_head": per_episode[:20],
        "examples": examples,
    }
    output = Path("/mnt/data/cyh/libero_core_lt_action_gripper_stats.json")
    output.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2)[:5000])
    print(f"saved={output}")


if __name__ == "__main__":
    main()
