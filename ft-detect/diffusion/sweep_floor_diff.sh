#!/usr/bin/env bash
# Phase 2 (diffusion) floor sweep, NO injection. There is no step watchdog to sweep, so the
# knobs are the two timeouts that exist: --scheduler-rpc-timeout (HTTP -> rank 0) and
# --dist-timeout (torch PG). Each run also records per-step durations per shape
# (steps_rank*.jsonl), which step_stats.py turns into the *hypothetical* step-watchdog
# floor per resolution: the central Phase 2 measurement.
#
#   RATE=<req/s> ./sweep_floor_diff.sh <mixed|harsh> [timeouts...]     # default 300 60 30 10 5 2
#
# Env: RATE (required), DURATION (default 1200), SWEEP rpc|dist|together (default rpc),
#      plus launch_diff.sh's MODEL / CUDA_VISIBLE_DEVICES / NGPU / PARALLEL_ARGS / FT_PORT
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/.." && pwd)
PROFILE=${1:?profile mixed|harsh}; shift
TIMEOUTS=${*:-"300 60 30 10 5 2"}
: "${RATE:?set RATE (req/s)}"
DURATION=${DURATION:-1200}
SWEEP=${SWEEP:-rpc}
RUNS_DIR=${RUNS_DIR:-$HERE/runs}
PYTHON=${PYTHON:-python}
export FT_API=diffusion
mkdir -p "$RUNS_DIR" "$HERE/results"

cleanup_run() {
  "$PYTHON" "$ROOT/stop.py" || true
  sleep 3
  [ -n "${LOAD_PID:-}" ] && kill -INT "$LOAD_PID" 2>/dev/null
  [ -n "${PROBE_PID:-}" ] && kill -INT "$PROBE_PID" 2>/dev/null
  wait 2>/dev/null
  LOAD_PID=""; PROBE_PID=""
}
trap 'echo "[sweep_diff] interrupted"; cleanup_run; exit 130' INT TERM

for T in $TIMEOUTS; do
  case "$SWEEP" in
    rpc)      export RPC_TIMEOUT="$T"; unset DIST_TIMEOUT ;;
    dist)     export DIST_TIMEOUT="$T"; unset RPC_TIMEOUT ;;
    together) export RPC_TIMEOUT="$T" DIST_TIMEOUT="$T" ;;
    *) echo "bad SWEEP=$SWEEP"; exit 2 ;;
  esac
  RUN="$RUNS_DIR/diff_floor_${PROFILE}_${SWEEP}_T${T}_$(date +%Y%m%d-%H%M%S)"
  export FT_RUN_DIR="$RUN"
  mkdir -p "$RUN/logs"
  echo "=============== [diffusion] floor profile=$PROFILE $SWEEP T=$T -> $RUN"
  if ! "$HERE/launch_diff.sh"; then
    echo "[sweep_diff] launch failed at T=$T"; touch "$RUN/launch_failed"; "$PYTHON" "$ROOT/stop.py" || true
    "$PYTHON" "$ROOT/analyze.py" "$RUN" --gap 60 --csv "$HERE/results/floor_${PROFILE}.csv" --md "$HERE/results/floor_${PROFILE}.md" || true
    continue
  fi
  "$PYTHON" "$ROOT/probe.py" --gen-timeout 60 > "$RUN/logs/probe.out" 2>&1 & PROBE_PID=$!
  "$PYTHON" "$HERE/load_diff.py" ${LOAD_ARGS:-} --rate "$RATE" --duration "$DURATION" --profile "$PROFILE" --drain-timeout 300 \
      > "$RUN/logs/load.out" 2>&1 & LOAD_PID=$!
  wait "$LOAD_PID"; LOAD_PID=""
  sleep 5
  cleanup_run
  "$PYTHON" "$ROOT/analyze.py" "$RUN" --gap 60 --csv "$HERE/results/floor_${PROFILE}.csv" --md "$HERE/results/floor_${PROFILE}.md" || true
  "$PYTHON" "$HERE/step_stats.py" "$RUN" --md "$RUN/step_stats.md" || true
  sleep 10
done
"$PYTHON" "$HERE/step_stats.py" "$RUNS_DIR"/diff_floor_"$PROFILE"_* --md "$HERE/results/step_stats_${PROFILE}.md" --csv "$HERE/results/step_stats_${PROFILE}.csv" || true
echo "[sweep_diff] done: $HERE/results/floor_${PROFILE}.md and step_stats_${PROFILE}.md"
