import re
import sys
from collections import defaultdict


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: parse_eval_log.py <log_path>")
        return 2

    path = sys.argv[1]
    counts = defaultdict(lambda: [0, 0])
    summary = []
    total_success = 0
    total_count = 0
    episode_re = re.compile(r"Task (\d+) .* episode \d+: success (True|False)")
    summary_re = re.compile(r"(Overall success rate|Task .* success rate)")

    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if summary_re.search(line):
                summary.append(line.rstrip())
            match = episode_re.search(line)
            if not match:
                continue
            task = int(match.group(1))
            ok = match.group(2) == "True"
            counts[task][0] += int(ok)
            counts[task][1] += 1
            total_success += int(ok)
            total_count += 1

    print(f"path={path}")
    print(f"episodes={total_count}")
    print(f"successes={total_success}")
    print(f"overall={total_success / total_count if total_count else 'NA'}")
    for task in range(10):
        success, count = counts[task]
        rate = success / count if count else "NA"
        print(f"task_{task}=successes:{success} episodes:{count} rate:{rate}")
    if summary:
        print("summary_lines_begin")
        for line in summary:
            print(line)
        print("summary_lines_end")
    else:
        print("summary_lines=none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
