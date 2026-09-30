#!/usr/bin/env bash
set -euo pipefail
ROOT=/bak_home/songcl/Project/AdaRES_project/AdaRES
exec 9>/bak_home/songcl/Project/AdaRES_project/.y15_gpu1_diagnostic.lock
flock -n 9 || exit 75
read -r USED UTIL < <(nvidia-smi -i 1 --query-gpu=memory.used,utilization.gpu \
  --format=csv,noheader,nounits | tr ',' ' ')
(( USED < 1024 && UTIL < 10 )) || exit 75
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT:$ROOT/ipdn/lib:/home/songcl/.conda/envs/adares_project/lib/python3.8/site-packages:/home/songcl/.local/lib/python3.8/site-packages/pointnet2-0.0.0-py3.8-linux-x86_64.egg"
cd "$ROOT"
PY=/home/songcl/.conda/envs/adares_project/bin/python
"$PY" tools/test_adares_y17_pairing.py --output y17_unit_20260930.json
timeout --signal=TERM --kill-after=10s 240s "$PY" -u tools/probe_adares_y17_entry.py \
  --entry tools/y17_proposed_train_20260930.py --output y17_entry_20260930.json
