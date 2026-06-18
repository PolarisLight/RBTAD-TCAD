#!/usr/bin/env bash
set -euo pipefail

chmod +x /mnt/data/cyh/train_tcad_final_full_23.sh
nohup bash /mnt/data/cyh/train_tcad_final_full_23.sh \
  >/mnt/data/cyh/tcad_final_full_launcher.log 2>&1 </dev/null &
pid=$!
echo "$pid" > /mnt/data/cyh/tcad_final_full_23.pid
echo "PID=$pid"
