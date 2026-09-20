#!/usr/bin/env bash
# Wait for AdaRES-B1 to finish, smoke-test AdaRES-B2, then launch it.
set -u

ROOT=/bak_home/songcl/Project/AdaRES_project/AdaRES
PY=/home/songcl/.conda/envs/adares_project/bin/python
SITE=/home/songcl/.conda/envs/adares_project/lib/python3.8/site-packages
EGG=/home/songcl/.local/lib/python3.8/site-packages/pointnet2-0.0.0-py3.8-linux-x86_64.egg
LOG="$ROOT/logs/watch_b1_to_b2.log"
cd "$ROOT" || exit 1

echo "[$(date '+%F %T')] watcher started" >> "$LOG"
B1_PID_FILE="$ROOT/logs/.pid_adares_b1"
if [[ ! -f "$B1_PID_FILE" ]]; then
  echo "[$(date '+%F %T')] missing B1 pid file" >> "$LOG"
  exit 1
fi
B1_PID=$(cat "$B1_PID_FILE")

while kill -0 "$B1_PID" 2>/dev/null; do
  sleep 60
done
echo "[$(date '+%F %T')] B1 pid $B1_PID exited" >> "$LOG"
sleep 30

for _ in $(seq 1 30); do
  if ! pgrep -f "tools/train.py" >/dev/null 2>&1; then
    break
  fi
  sleep 30
done
if pgrep -f "tools/train.py" >/dev/null 2>&1; then
  echo "[$(date '+%F %T')] training processes still present; abort" >> "$LOG"
  exit 1
fi

printf '%s\n' \
  '# AdaRES-B2: metric-calibrated indicator' \
  'from ipdn.model_paper5.adares_b2_metric_indicator import IPDN' \
  > "$ROOT/ipdn/__init__.py"

export PYTHONPATH="$ROOT:$ROOT/ipdn/lib:$SITE:$EGG"

SMOKE_DIR="$ROOT/../AdaRES_checkpoints/smoke_adares_b2_after_b1"
rm -rf "$SMOKE_DIR"
echo "[$(date '+%F %T')] running B2 smoke" >> "$LOG"
if ! "$PY" -S "$ROOT/tools/smoke_adares_b0.py" \
    "$ROOT/configs_song/model_paper5/adares_b2_metric_indicator_res.yaml" \
    --output-dir "$SMOKE_DIR" --gpu-id 0 --scenes 1 --seed 1999 \
    >> "$LOG" 2>&1; then
  echo "[$(date '+%F %T')] B2 smoke failed; not launching" >> "$LOG"
  exit 1
fi
echo "[$(date '+%F %T')] B2 smoke passed; launching training" >> "$LOG"

TS=$(date +%Y%m%d_%H%M%S)
TRAIN_LOG="$ROOT/logs/train_adares_b2_${TS}.log"
nohup "$PY" -S "$ROOT/tools/train.py" \
  "$ROOT/configs_song/model_paper5/adares_b2_metric_indicator_res.yaml" \
  --dist --num_gpus 2 --gpu_ids 0 1 \
  > "$TRAIN_LOG" 2>&1 < /dev/null &
TRAIN_PID=$!
echo "$TRAIN_PID" > "$ROOT/logs/.pid_adares_b2"
echo "[$(date '+%F %T')] B2 launched pid=$TRAIN_PID log=$TRAIN_LOG" >> "$LOG"
