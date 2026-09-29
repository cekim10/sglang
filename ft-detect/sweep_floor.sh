#!/usr/bin/env bash
# Step 3 driver: timeout-floor sweep with NO injection.
#
#   RATE=<req/s> ./sweep_floor.sh <mixed|harsh> [timeouts...]      # default 300 60 30 10 5 2
#
# Env:
#   RATE        (required) open-loop rate (same as Step 2)
#   DURATION    s per setting (default 1200 = 20 min)
#   SWEEP       together | watchdog | dist   (default together)
#               together: --watchdog-timeout T --dist-timeout T
#               watchdog: only --watchdog-timeout T
#               dist:     only --dist-timeout T
#   RUNS_DIR, PYTHON, plus launch.sh's MODEL / CUDA_VISIBLE_DEVICES / TP / FT_PORT
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
PROFILE=${1:?profile mixed|harsh}; shift
TIMEOUTS=${*:-"300 60 30 10 5 2"}
: "${RATE:?set RATE (req/s)}"
DURATION=${DURATION:-1200}
SWEEP=${SWEEP:-together}
RUNS_DIR=${RUNS_DIR:-$HERE/runs}
PYTHON=${PYTHON:-python}
mkdir -p "$RUNS_DIR" "$HERE/results"

cleanup_run() {
  "$PYTHON" "$HERE/stop.py" || true
  sleep 3
  [ -n "${LOAD_PID:-}" ] && kill -INT "$LOAD_PID" 2>/dev/null
  [ -n "${PROBE_PID:-}" ] && kill -INT "$PROBE_PID" 2>/dev/null
  wait 2>/dev/null
  LOAD_PID=""; PROBE_PID=""
}
trap 'echo "[sweep] interrupted"; cleanup_run; exit 130' INT TERM

for T in $TIMEOUTS; do
  case "$SWEEP" in
    together) export WATCHDOG_TIMEOUT="$T" DIST_TIMEOUT="$T" ;;
    watchdog) export WATCHDOG_TIMEOUT="$T"; unset DIST_TIMEOUT ;;
    dist)     export DIST_TIMEOUT="$T"; unset WATCHDOG_TIMEOUT ;;
    *) echo "bad SWEEP=$SWEEP"; exit 2 ;;
  esac
  RUN="$RUNS_DIR/floor_${PROFILE}_${SWEEP}_T${T}_$(date +%Y%m%d-%H%M%S)"
  export FT_RUN_DIR="$RUN"
  mkdir -p "$RUN/logs"
  echo "=============== floor sweep profile=$PROFILE $SWEEP T=$T -> $RUN"
  if ! "$HERE/launch.sh"; then
    echo "[sweep] launch failed at T=$T (itself a finding: startup collectives exceed the timeout?)"
    touch "$RUN/launch_failed"; "$PYTHON" "$HERE/stop.py" || true
    "$PYTHON" "$HERE/analyze.py" "$RUN" --csv "$HERE/results/floor_${PROFILE}.csv" --md "$HERE/results/floor_${PROFILE}.md" || true
    continue
  fi
  "$PYTHON" "$HERE/probe.py" > "$RUN/logs/probe.out" 2>&1 & PROBE_PID=$!
  "$PYTHON" "$HERE/load.py" --rate "$RATE" --duration "$DURATION" --profile "$PROFILE" --drain-timeout 120 \
      > "$RUN/logs/load.out" 2>&1 & LOAD_PID=$!
  wait "$LOAD_PID"; LOAD_PID=""
  sleep 5
  cleanup_run
  "$PYTHON" "$HERE/analyze.py" "$RUN" --csv "$HERE/results/floor_${PROFILE}.csv" --md "$HERE/results/floor_${PROFILE}.md" || true
  sleep 10
done
echo "[sweep] done: $HERE/results/floor_${PROFILE}.md"
