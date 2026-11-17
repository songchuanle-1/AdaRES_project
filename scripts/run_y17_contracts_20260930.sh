#!/usr/bin/env bash
set -euo pipefail
ROOT=/bak_home/songcl/Project/AdaRES_project/AdaRES
MODE="${1:?controls or stress}"
exec 9>/bak_home/songcl/Project/AdaRES_project/.y15_gpu1_diagnostic.lock
flock -n 9 || { echo GPU_LEASE_BUSY; exit 75; }
read -r USED UTIL < <(nvidia-smi -i 1 --query-gpu=memory.used,utilization.gpu \
  --format=csv,noheader,nounits | tr ',' ' ')
(( USED < 1024 && UTIL < 10 )) || { echo GPU_NOT_IDLE; exit 75; }
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT:$ROOT/ipdn/lib:/home/songcl/.conda/envs/adares_project/lib/python3.8/site-packages:/home/songcl/.local/lib/python3.8/site-packages/pointnet2-0.0.0-py3.8-linux-x86_64.egg"
cd "$ROOT"
MANIFEST=y17_train_manifest_20260930.json
[[ "$MODE" != stress ]] || MANIFEST=y15_stress_manifest_20260930.json
timeout --signal=TERM --kill-after=10s 240s \
  /home/songcl/.conda/envs/adares_project/bin/python -u \
  tools/probe_adares_y17_contracts.py --gpu-id 1 --mode "$MODE" \
  --config configs_song/model_paper5/adares_y17_target_equivalent_pairing_res.yaml \
  --manifest "$MANIFEST" --output "y17_physical_${MODE}_20260930.json"
