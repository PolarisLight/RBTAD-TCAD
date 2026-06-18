#!/usr/bin/env python3
import re
import sys
from collections import defaultdict

path = sys.argv[1]
counts = defaultdict(int)
success = defaultdict(int)
last_ts = None
pat = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*Task (\d+) .* episode \d+: success (True|False)")

with open(path, "r", encoding="utf-8", errors="replace") as handle:
    for line in handle:
        m = pat.search(line)
        if not m:
            continue
        last_ts = m.group(1)
        task = int(m.group(2))
        ok = m.group(3) == "True"
        counts[task] += 1
        success[task] += int(ok)

total = sum(counts.values())
print(f"completed={total}/300 last_ts={last_ts}")
for task in range(10):
    c = counts[task]
    rate = success[task] / c if c else 0.0
    print(f"task{task}: {c}/30 success={success[task]} rate={rate:.3f}")
