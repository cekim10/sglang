#!/usr/bin/env bash
# Launch a two-rank standalone probe (rank 0 on the first GPU, rank 1 on the second), wait for
# rank 0, then SIGCONT/kill rank 1. Usage: ./run_pair.sh <script.py> [args passed to both ranks]
# Env: CUDA_VISIBLE_DEVICES=A,B ; MAX_S (default 300) hard limit for rank 0.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
SCRIPT=${1:?script}; shift
PYTHON=${PYTHON:-python}
MAX_S=${MAX_S:-300}
IFS=, read -r GPU0 GPU1 <<< "${CUDA_VISIBLE_DEVICES:-0,1}"
PORT=$(( 29600 + RANDOM % 300 ))
OUT=${OUT_DIR:-$HERE/results/pair_$(basename "$SCRIPT" .py)}; mkdir -p "$OUT"
( export MASTER_ADDR=127.0.0.1 MASTER_PORT=$PORT CUDA_VISIBLE_DEVICES=$GPU1
  exec "$PYTHON" "$HERE/$SCRIPT" --rank 1 --out "$OUT/rank1.json" "$@" > "$OUT/rank1.log" 2>&1 ) &
R1=$!
( export MASTER_ADDR=127.0.0.1 MASTER_PORT=$PORT CUDA_VISIBLE_DEVICES=$GPU0
  exec timeout "$MAX_S" "$PYTHON" "$HERE/$SCRIPT" --rank 0 --out "$OUT/rank0.json" "$@" 2>&1 | tee "$OUT/rank0.log" ) &
R0=$!
wait $R0
kill -CONT $R1 2>/dev/null; sleep 1; kill -9 $R1 2>/dev/null; wait $R1 2>/dev/null
pkill -9 -f "$SCRIPT --rank 1" 2>/dev/null
echo "[run_pair] outputs in $OUT"
