#!/usr/bin/env bash
# Launch sp_shrink_probe.py ranks on this machine without a supervisor (a stopped rank must not make
# an agent tear the survivors down). One invocation per machine.
#   single machine, CPU:   BACKEND=gloo SHAPE=cpu NPROC=4 ./run_shrink.sh
#   two machines, GPUs:    elves-01: NODE_RANK=0 NNODES=2 NPROC=2 MASTER_ADDR=elves-01.be.ucsc.edu ./run_shrink.sh
#                          elves-03: NODE_RANK=1 NNODES=2 NPROC=2 MASTER_ADDR=elves-01.be.ucsc.edu ./run_shrink.sh
# Env: BACKEND nccl|gloo, SHAPE wan|small|cpu, STEPS 10, FAIL_STEP 4, DEADLINE_S 5, MAX_S 300, MASTER_PORT 29561,
#      OUT_DIR (default results/shrink_<backend>_<shape>). For two machines also export the network
#      settings that made xnode_a2a_check.py work (LD_PRELOAD=.../mss_clamp.so NCCL_IB_DISABLE=1
#      NCCL_SOCKET_IFNAME=... GLOO_SOCKET_IFNAME=... NCCL_P2P_DISABLE=1). gloo resolves the short
#      hostname otherwise, which Ubuntu maps to 127.0.1.1, and the other machine cannot connect.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
BACKEND=${BACKEND:-nccl}; SHAPE=${SHAPE:-wan}; NPROC=${NPROC:-2}; NNODES=${NNODES:-1}; NODE_RANK=${NODE_RANK:-0}
WORLD=$(( NPROC * NNODES ))
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1} MASTER_PORT=${MASTER_PORT:-29561} WORLD_SIZE=$WORLD
OUT=${OUT_DIR:-$HERE/results/shrink_${BACKEND}_${SHAPE}}; mkdir -p "$OUT"
rm -f "$OUT"/rank*.log "$OUT"/rank*.json      # stale logs from an earlier run would be printed below
echo "[run_shrink] $(hostname): NODE_RANK=$NODE_RANK of NNODES=$NNODES (node 0 must run on the MASTER_ADDR machine: $MASTER_ADDR)"
PYTHON=${PYTHON:-python}
pids=()
for l in $(seq 0 $(( NPROC - 1 ))); do
  r=$(( NODE_RANK * NPROC + l ))
  RANK=$r LOCAL_RANK=$l "$PYTHON" "$HERE/sp_shrink_probe.py" --backend "$BACKEND" --shape "$SHAPE" \
      --steps "${STEPS:-10}" --fail-step "${FAIL_STEP:-4}" --deadline-s "${DEADLINE_S:-5}" --max-s "${MAX_S:-300}" --out "$OUT/rank$r.json" \
      > "$OUT/rank$r.log" 2>&1 &
  pids+=($!)
done
echo "[run_shrink] node $NODE_RANK: ranks $(( NODE_RANK * NPROC ))..$(( NODE_RANK * NPROC + NPROC - 1 )) of $WORLD; logs in $OUT"
for p in "${pids[@]}"; do wait "$p"; done
grep -h -E "gloo_check|references|deadline_miss|abort_done|membership|survivor_group|result|fenced|watchdog|Error|error" "$OUT"/rank*.log | cut -c1-600
