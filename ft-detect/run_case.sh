#!/usr/bin/env bash
# Step 2 driver: N runs of one injection case, fresh server each time.
#
#   RATE=<req/s from calibrate.sh> ./run_case.sh <A|B> <target_rank> <n_runs>
#
# Env (besides launch.sh's MODEL / CUDA_VISIBLE_DEVICES / TP / FT_PORT / WATCHDOG_TIMEOUT / DIST_TIMEOUT):
#   RATE              (required) open-loop request rate
#   PROFILE           mixed | harsh          (default mixed)
#   RUNS_DIR          where per-run dirs go  (default ./runs)
#   MIN_WARMUP        s of load before checking steady state (default 90)
#   STEADY_WAIT       max s to wait for steady state (default 600)
#   MAX_DETECT_WAIT   max s to wait for an engine-side detection after injection (default 900;
#                     stock watchdog fires at 300-450 s, torch NCCL default is 600 s)
#   POST_GRACE        s to keep observing after detection (default 30)
#   PYTHON            interpreter
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
CASE=${1:?case A|B}; RANK=${2:?target rank}; N=${3:?n_runs}
: "${RATE:?set RATE (req/s) from calibrate.sh}"
PROFILE=${PROFILE:-mixed}
RUNS_DIR=${RUNS_DIR:-$HERE/runs}
MIN_WARMUP=${MIN_WARMUP:-90}
STEADY_WAIT=${STEADY_WAIT:-600}
MAX_DETECT_WAIT=${MAX_DETECT_WAIT:-900}
POST_GRACE=${POST_GRACE:-30}
PYTHON=${PYTHON:-python}
LOAD_DURATION=$(( MIN_WARMUP + STEADY_WAIT + MAX_DETECT_WAIT + POST_GRACE + 120 ))
mkdir -p "$RUNS_DIR" "$HERE/results"

DETECT_RE='"pattern":"(watchdog_fire|scheduler_exception|subprocess_crashed|sigquit|kill_tree|nccl_timeout|nccl_error|abort|torch_dist_error|cuda_error|scheduler_terminated)"|"who":"http_server","pid":[0-9]+,"state":"(dead|zombie)"'

cleanup_run() {
  # Order matters: tear the server down first so hung requests fail on their own
  # (connection reset) and are logged as such; only then stop load and probe.
  "$PYTHON" "$HERE/stop.py" || true
  sleep 3
  [ -n "${LOAD_PID:-}" ] && kill -INT "$LOAD_PID" 2>/dev/null
  [ -n "${PROBE_PID:-}" ] && kill -INT "$PROBE_PID" 2>/dev/null
  wait 2>/dev/null
  LOAD_PID=""; PROBE_PID=""
}
trap 'echo "[run_case] interrupted"; cleanup_run; exit 130' INT TERM

for i in $(seq 1 "$N"); do
  RUN="$RUNS_DIR/${CASE}_r${RANK}_${i}_$(date +%Y%m%d-%H%M%S)"
  export FT_RUN_DIR="$RUN"
  mkdir -p "$RUN/logs"
  echo "=============== case $CASE rank $RANK run $i/$N -> $RUN"
  if ! "$HERE/launch.sh"; then
    echo "[run_case] launch failed; recorded and moving on"; touch "$RUN/launch_failed"; "$PYTHON" "$HERE/stop.py" || true; continue
  fi
  "$PYTHON" "$HERE/recon.py" --json "$RUN/recon.json" > "$RUN/logs/recon.out" 2>&1 || true

  "$PYTHON" "$HERE/probe.py" > "$RUN/logs/probe.out" 2>&1 & PROBE_PID=$!
  "$PYTHON" "$HERE/load.py" --rate "$RATE" --duration "$LOAD_DURATION" --profile "$PROFILE" --tag "$i" \
      > "$RUN/logs/load.out" 2>&1 & LOAD_PID=$!
  echo "[run_case] probe pid $PROBE_PID, load pid $LOAD_PID; warming up ${MIN_WARMUP}s"
  sleep "$MIN_WARMUP"
  if ! "$PYTHON" "$HERE/steady.py" --wait "$STEADY_WAIT"; then
    echo "[run_case] WARNING: steady state not reached within ${STEADY_WAIT}s; injecting anyway (flagged in summary)"
    touch "$RUN/not_steady"
  fi

  n_before=$(wc -l < "$RUN/probe.jsonl" 2>/dev/null || echo 0)
  "$PYTHON" "$HERE/inject.py" --case "$CASE" --rank "$RANK" --run-id "$(basename "$RUN")" || { echo "[run_case] inject failed"; cleanup_run; continue; }

  t0=$(date +%s); detected=0
  while [ $(( $(date +%s) - t0 )) -lt "$MAX_DETECT_WAIT" ]; do
    if tail -n +"$(( n_before + 1 ))" "$RUN/probe.jsonl" 2>/dev/null | grep -Eq "$DETECT_RE"; then
      detected=1; echo "[run_case] engine-side detection after $(( $(date +%s) - t0 )) s"; break
    fi
    sleep 1
  done
  [ "$detected" = 1 ] || { echo "[run_case] no engine-side detection within ${MAX_DETECT_WAIT}s"; touch "$RUN/no_engine_detect"; }
  sleep "$POST_GRACE"

  cleanup_run
  "$PYTHON" "$HERE/analyze.py" "$RUN" --csv "$HERE/results/case${CASE}.csv" --md "$HERE/results/case${CASE}.md" || true
  sleep 10   # let the GPUs free up before the next launch
done
echo "[run_case] done: $HERE/results/case${CASE}.md"
