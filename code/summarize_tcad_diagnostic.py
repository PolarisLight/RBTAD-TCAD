import json
import sys
from collections import defaultdict

import numpy as np


path = sys.argv[1]
data = json.load(open(path, "r", encoding="utf-8"))
rows = data["rows"]
groups = defaultdict(list)
for row in rows:
    prefix = row["instruction"]
    groups[prefix].append(row["margin"])

print("SUMMARY", json.dumps(data["summary"], indent=2))
print("GROUPS")
for instruction, margins in sorted(groups.items()):
    arr = np.asarray(margins, dtype=np.float32)
    print(
        json.dumps(
            {
                "n": int(len(arr)),
                "mean": float(arr.mean()),
                "positive_rate": float((arr > 0).mean()),
                "instruction": instruction,
            },
            ensure_ascii=False,
        )
    )
