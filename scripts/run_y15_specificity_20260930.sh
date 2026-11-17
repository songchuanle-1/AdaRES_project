#!/usr/bin/env bash
set -euo pipefail
ROOT=/bak_home/songcl/Project/AdaRES_project/AdaRES
exec 9>/bak_home/songcl/Project/AdaRES_project/.y15_gpu0_diagnostic.lock
flock -n 9 || { echo GPU_LEASE_BUSY; exit 75; }
read -r USED UTIL < <(nvidia-smi -i 0 --query-gpu=memory.used,utilization.gpu \
  --format=csv,noheader,nounits | tr ',' ' ')
if (( USED >= 1024 || UTIL >= 10 )); then
  echo "GPU_NOT_IDLE memory=$USED utilization=$UTIL"
  exit 75
fi
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT:$ROOT/ipdn/lib:/home/songcl/.conda/envs/adares_project/lib/python3.8/site-packages:/home/songcl/.local/lib/python3.8/site-packages/pointnet2-0.0.0-py3.8-linux-x86_64.egg"
cd "$ROOT"
timeout --signal=TERM --kill-after=10s 240s \
  /home/songcl/.conda/envs/adares_project/bin/python -u \
  tools/probe_adares_y15_specificity.py --gpu-id 0 \
  --config configs_song/model_paper5/adares_y15_query_completion_res.yaml \
  --source y15_specificity_source_20260930.json \
  --output y15_specificity_20260930.json
