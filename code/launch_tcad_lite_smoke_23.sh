#!/usr/bin/env bash
set -euo pipefail

chmod +x /mnt/data/cyh/train_tcad_lite_smoke_23.sh
nohup /usr/bin/env bash /mnt/data/cyh/train_tcad_lite_smoke_23.sh \
  >/mnt/data/cyh/tcad_lite_smoke_launcher.log 2>&1 </dev/null &
pid=$!
echo "$pid" > /mnt/data/cyh/tcad_lite_smoke_23.pid
echo "PID=$pid"
