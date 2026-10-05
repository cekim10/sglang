#!/usr/bin/env bash
# Option-3 probe runner: two ranks on two GPUs, rank 1 freezes, rank 0 either floods launches
# behind the stuck collective or holds back and polls. Prints a summary per mode.
#
#   ./run_launch_discipline.sh [flood|disciplined|both]      (default both)
# Env: CUDA_VISIBLE_DEVICES=0,1 (two GPUs), NCCL_P2P_DISABLE as for the host, DEADLINE_S (10), MAX_S (120)
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
MODES=${1:-both}; [ "$MODES" = both ] && MODES="flood disciplined"
PYTHON=${PYTHON:-python}
DEADLINE_S=${DEADLINE_S:-10}
MAX_S=${MAX_S:-120}
OUT=${RESULTS_DIR:-$HERE/results}/launch_discipline; mkdir -p "$OUT"
IFS=, read -r GPU0 GPU1 <<< "${CUDA_VISIBLE_DEVICES:-0,1}"
for MODE in $MODES; do
  PORT=$(( 29600 + RANDOM % 300 ))
  echo "=============== launch discipline probe: mode=$MODE GPUs=$GPU0,$GPU1 port=$PORT"
  ( export MASTER_ADDR=127.0.0.1 MASTER_PORT=$PORT CUDA_VISIBLE_DEVICES=$GPU1
    exec "$PYTHON" "$HERE/launch_discipline_probe.py" --rank 1 --mode "$MODE" --max-s "$MAX_S" --out "$OUT/${MODE}_rank1.json" > "$OUT/${MODE}_rank1.log" 2>&1 ) &
  R1=$!
  ( export MASTER_ADDR=127.0.0.1 MASTER_PORT=$PORT CUDA_VISIBLE_DEVICES=$GPU0
    exec "$PYTHON" "$HERE/launch_discipline_probe.py" --rank 0 --mode "$MODE" --deadline-s "$DEADLINE_S" --max-s "$MAX_S" --out "$OUT/${MODE}_rank0.json" 2>&1 | tee "$OUT/${MODE}_rank0.log" | grep -v "side\.\(launch_begin\|launch_done\)" ) &
  R0=$!
  wait $R0
  kill -CONT $R1 2>/dev/null; sleep 1; kill -9 $R1 2>/dev/null; wait $R1 2>/dev/null
  pkill -9 -f "launch_discipline_probe.py --rank 1" 2>/dev/null
  "$PYTHON" "$HERE/launch_discipline_summary.py" "$OUT/${MODE}_rank0.json"
  sleep 3
done
echo "[launch discipline] outputs in $OUT"
