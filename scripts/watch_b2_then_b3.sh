#!/usr/bin/env bash
# Wait for B2, then launch B3 only if the Phase 1 target is still unmet.
set -u

ROOT=/bak_home/songcl/Project/AdaRES_project/AdaRES
PY=/home/songcl/.conda/envs/project_cluster/bin/python
SITE=/home/songcl/.conda/envs/project_cluster/lib/python3.8/site-packages
EGG=/home/songcl/.local/lib/python3.8/site-packages/pointnet2-0.0.0-py3.8-linux-x86_64.egg
WATCH_LOG="$ROOT/logs/watch_b2_to_b3.log"
B2_PID_FILE="$ROOT/logs/.pid_adares_b2"
TARGET=0.515

cd "$ROOT" || exit 1
echo "[$(date '+%F %T')] watcher started" >> "$WATCH_LOG"

if [[ ! -f "$B2_PID_FILE" ]]; then
  echo "[$(date '+%F %T')] missing B2 pid file" >> "$WATCH_LOG"
  exit 1
fi
B2_PID=$(cat "$B2_PID_FILE")

while kill -0 "$B2_PID" 2>/dev/null; do
  sleep 60
done
echo "[$(date '+%F %T')] B2 pid $B2_PID exited" >> "$WATCH_LOG"

for _ in $(seq 1 30); do
  if ! pgrep -f "[t]ools/train.py" >/dev/null 2>&1; then
    break
  fi
  sleep 30
done
if pgrep -f "[t]ools/train.py" >/dev/null 2>&1; then
  echo "[$(date '+%F %T')] residual training process; abort" >> "$WATCH_LOG"
  exit 1
fi

B2_LOG=$(ls -t "$ROOT"/logs/train_adares_b2_* 2>/dev/null | head -1)
if [[ -z "${B2_LOG:-}" ]]; then
  echo "[$(date '+%F %T')] B2 log missing; abort" >> "$WATCH_LOG"
  exit 1
fi
BEST=$(tr '\r' '\n' < "$B2_LOG" |
  awk '/^mIOU:/ {gsub(/\./, "", $2); value=$2/1000; if (value>best) best=value} END {printf "%.3f", best+0}')
echo "[$(date '+%F %T')] B2 best mIoU=$BEST target=$TARGET" >> "$WATCH_LOG"

if awk -v best="$BEST" -v target="$TARGET" 'BEGIN {exit !(best >= target)}'; then
  echo "[$(date '+%F %T')] target reached; B3 will not launch" >> "$WATCH_LOG"
  exit 0
fi

printf '%s\n' \
  '# AdaRES-B3: mask-evidence indicator' \
  'from ipdn.model_paper5.adares_b3_mask_evidence import IPDN' \
  > "$ROOT/ipdn/__init__.py"

export PYTHONPATH="$ROOT:$ROOT/ipdn/lib:$SITE:$EGG"
TS=$(date +%Y%m%d_%H%M%S)
TRAIN_LOG="$ROOT/logs/train_adares_b3_${TS}.log"
nohup "$PY" -S "$ROOT/tools/train.py" \
  "$ROOT/configs_song/model_paper5/adares_b3_mask_evidence_res.yaml" \
  --dist --num_gpus 2 --gpu_ids 0 1 \
  > "$TRAIN_LOG" 2>&1 < /dev/null &
TRAIN_PID=$!
echo "$TRAIN_PID" > "$ROOT/logs/.pid_adares_b3"
echo "[$(date '+%F %T')] B3 launched pid=$TRAIN_PID log=$TRAIN_LOG" >> "$WATCH_LOG"
