#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/data/cyh/VLA-long-tail
UPLOAD=/mnt/data/cyh/anchor_upload_tmp
SMOKE_ID=anchor_rbtad_smoke_l2all_w125_a005_tail9_s5_seed7_b20
RUN_ID=anchor_rbtad_l2all_w125_a005_tail9_s500_seed7_b20
FORMAL_LOG=/mnt/data/cyh/${RUN_ID}.log

echo "== install Anchor-RBTAD patch $(date -Is) =="
cp "${UPLOAD}/train.py" "${ROOT}/vla_scripts/train.py"
cp "${UPLOAD}/base_strategy.py" "${ROOT}/prismatic/training/strategies/base_strategy.py"
cp "${UPLOAD}/run_anchor_rbtad_23.sh" /mnt/data/cyh/run_anchor_rbtad_23.sh
cp "${UPLOAD}/eval_sabre_step1000_23.sh" /mnt/data/cyh/eval_sabre_step1000_23.sh
chmod +x /mnt/data/cyh/run_anchor_rbtad_23.sh /mnt/data/cyh/eval_sabre_step1000_23.sh

cd "${ROOT}"
/mnt/data/cyh/envs/vla-long-tail/bin/python -m py_compile \
  vla_scripts/train.py \
  prismatic/training/strategies/base_strategy.py

echo "== smoke Anchor-RBTAD $(date -Is) =="
RUN_ID="${SMOKE_ID}" LIMIT_STEPS=5 DO_EVAL=0 bash /mnt/data/cyh/run_anchor_rbtad_23.sh
SMOKE_LOG=/mnt/data/cyh/${SMOKE_ID}.log
tail -120 "${SMOKE_LOG}" || true

SMOKE_DEBUG="${ROOT}/runs/anchor_rbtad/${SMOKE_ID}/tcad-debug.csv"
if [[ ! -f "${SMOKE_DEBUG}" ]]; then
  echo "missing smoke debug file: ${SMOKE_DEBUG}"
  exit 3
fi
if ! head -1 "${SMOKE_DEBUG}" | grep -q "anchor_l2_loss"; then
  echo "anchor_l2_loss column missing"
  exit 4
fi
if ! grep -q "Anchor L2 enabled for" "${SMOKE_LOG}"; then
  echo "Anchor L2 enable log missing"
  exit 5
fi
if grep -q "Anchor L2 enabled for 0 tensors" "${SMOKE_LOG}"; then
  echo "Anchor L2 matched zero tensors"
  exit 6
fi

echo "== launch Anchor-RBTAD formal 500-step + eval $(date -Is) =="
if ps -u cyh -o cmd | grep -E "run_anchor_rbtad_23.sh|parallel_libero_evaluator_egl.py.*anchor_rbtad" | grep -v grep >/dev/null; then
  echo "Anchor-RBTAD job already running; not launching duplicate"
else
  RUN_ID="${RUN_ID}" LIMIT_STEPS=500 DO_EVAL=1 nohup bash /mnt/data/cyh/run_anchor_rbtad_23.sh > "${FORMAL_LOG}" 2>&1 &
  echo "formal_pid=$!"
fi

echo "== status $(date -Is) =="
ps -u cyh -o pid,etime,cmd | grep -E "run_anchor_rbtad_23.sh|parallel_libero_evaluator_egl.py.*anchor_rbtad" | grep -v grep || true
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true
tail -80 "${FORMAL_LOG}" 2>/dev/null || true
