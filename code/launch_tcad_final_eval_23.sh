#!/usr/bin/env bash
set -euo pipefail

chmod +x /mnt/data/cyh/eval_tcad_final_step34075_egl_23.sh
nohup bash /mnt/data/cyh/eval_tcad_final_step34075_egl_23.sh \
  >/mnt/data/cyh/tcad_final_eval_launcher.log 2>&1 </dev/null &
pid=$!
echo "$pid" > /mnt/data/cyh/tcad_final_eval_23.pid
echo "PID=$pid"
