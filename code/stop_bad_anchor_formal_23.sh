#!/usr/bin/env bash
set -euo pipefail

RUN_ID="anchor_rbtad_l2all_w125_a005_tail9_s500_seed7_b20"

echo "== stop invalid Anchor-RBTAD formal $(date -Is) =="
/mnt/data/cyh/envs/vla-long-tail/bin/python - <<'PY'
import os
import signal
import subprocess

run_id = "anchor_rbtad_l2all_w125_a005_tail9_s500_seed7_b20"
out = subprocess.check_output(["ps", "-u", "cyh", "-o", "pid=", "-o", "cmd="], text=True)
pids = []
for line in out.splitlines():
    line = line.strip()
    if not line:
        continue
    pid_s, _, cmd = line.partition(" ")
    if run_id in cmd and "stop_bad_anchor_formal_23.sh" not in cmd:
        pids.append(int(pid_s))
for pid in pids:
    print(f"killing {pid}")
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
print(f"killed={len(pids)}")
PY

sleep 5
echo "== remaining =="
/mnt/data/cyh/envs/vla-long-tail/bin/python - <<'PY'
import subprocess

run_id = "anchor_rbtad_l2all_w125_a005_tail9_s500_seed7_b20"
out = subprocess.check_output(["ps", "-u", "cyh", "-o", "pid=", "-o", "etime=", "-o", "cmd="], text=True)
for line in out.splitlines():
    if run_id in line and "stop_bad_anchor_formal_23.sh" not in line:
        print(line)
PY

echo "== gpu =="
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
