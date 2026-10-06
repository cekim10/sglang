#!/usr/bin/env bash
# Node-1 side of a two-node Test 3 (SP=4 -> SP=3). Run this on the worker machine first, then
# run_contain.sh on node 0 with the same settings. Each iteration launches this node's workers,
# waits until node 0's store (DIST_INIT_ADDR) has come up and gone away again (node 0 finished its
# run and stopped its server), stops the local workers, and starts over for the next run.
#
#   NNODES=2 NODE_RANK=1 NGPU=4 DIST_INIT_ADDR=elves-01.be.ucsc.edu:29600 N=3 ./run_contain_worker.sh
#
# Same env as run_contain.sh MODE=ours (MODEL, SHAPE, STEPS, FAIL_STEP, FAIL_RANK, DEADLINE_S,
# PARALLEL_ARGS, ENCODER_PARALLEL, network settings): the failing rank lives here, so the injection
# and containment settings must match node 0's.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/.." && pwd)
: "${DIST_INIT_ADDR:?set DIST_INIT_ADDR=host:port of node 0}"
N=${N:-1}; SHAPE=${SHAPE:-832x480x81}; FAIL_STEP=${FAIL_STEP:-4}; FAIL_RANK=${FAIL_RANK:-3}; DEADLINE_S=${DEADLINE_S:-5}
ENCODER_PARALLEL=${ENCODER_PARALLEL:-replicate}
export NNODES=${NNODES:-2} NODE_RANK=${NODE_RANK:-1} FT_API=diffusion
export FT_CONTAIN=1 FT_CONTAIN_DEADLINE_S="$DEADLINE_S" FT_FAIL_STEP="$FAIL_STEP" FT_FAIL_RANK="$FAIL_RANK" FT_FAIL_REQ=3 FT_FAIL_SHAPE="$SHAPE"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-0}"
export EXTRA_ARGS="${EXTRA_ARGS:-} --encoder-parallel $ENCODER_PARALLEL"
RUNS_DIR=${RUNS_DIR:-$HERE/runs}; PYTHON=${PYTHON:-python}
HOST=${DIST_INIT_ADDR%:*}; PORT=${DIST_INIT_ADDR##*:}
port_open() { "$PYTHON" -c "import socket,sys; s=socket.socket(); s.settimeout(1); sys.exit(0 if s.connect_ex(('$HOST',$PORT))==0 else 1)"; }
trap 'echo "[worker] interrupted"; "$PYTHON" "$ROOT/stop.py" || true; exit 130' INT TERM

for i in $(seq 1 "$N"); do
  RUN="$RUNS_DIR/contain_worker_$(date +%Y%m%d-%H%M%S)"; export FT_RUN_DIR="$RUN"; mkdir -p "$RUN/logs"
  echo "=============== [worker] run $i/$N node $NODE_RANK -> $RUN"
  "$HERE/launch_diff.sh" || { echo "[worker] launch failed"; "$PYTHON" "$ROOT/stop.py" || true; continue; }
  MAIN=$("$PYTHON" -c "import json;print(json.load(open('$RUN/pids.json'))['http_server'])")
  # node 0 may open and close the rendezvous port while it starts; only a store that stays closed
  # for CLOSED_S after having been up means node 0 finished its run
  seen=0; closed_since=""
  while kill -0 "$MAIN" 2>/dev/null; do
    if port_open; then
      [ "$seen" = 0 ] && echo "[worker] node 0 store is up"
      seen=1; closed_since=""
    elif [ "$seen" = 1 ]; then
      [ -z "$closed_since" ] && { closed_since=$(date +%s); echo "[worker] node 0 store not answering; waiting ${CLOSED_S:-60}s"; }
      if [ $(( $(date +%s) - closed_since )) -ge "${CLOSED_S:-60}" ]; then echo "[worker] node 0 store closed for ${CLOSED_S:-60}s: run finished"; break; fi
    fi
    sleep 2
  done
  kill -0 "$MAIN" 2>/dev/null || echo "[worker] local launcher exited on its own"
  grep -h "FT_CONTAIN\|FT_FAIL" "$RUN"/logs/rank*.log | cut -c1-300
  "$PYTHON" "$ROOT/stop.py" || true
  sleep 5
done
