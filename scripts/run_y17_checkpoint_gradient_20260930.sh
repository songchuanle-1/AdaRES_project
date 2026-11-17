#!/usr/bin/env bash
set -euo pipefail
ROOT=/bak_home/songcl/Project/AdaRES_project/AdaRES
exec 9>/bak_home/songcl/Project/AdaRES_project/.y15_gpu1_diagnostic.lock
flock -n 9 || { echo GPU_LEASE_BUSY; exit 75; }
read -r USED UTIL < <(nvidia-smi -i 1 --query-gpu=memory.used,utilization.gpu \
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
  tools/probe_adares_y17_native_gradient.py --gpu-id 1 \
  --config configs_song/model_paper5/adares_w2_pointmass_competition_res.yaml \
  --checkpoint ../AdaRES_checkpoints/adares_w2_pointmass_competition_res/20260926_145041/best.pth \
  --manifest y17_train_manifest_20260930.json \
  --output y17_checkpoint_gradient_20260930.json
