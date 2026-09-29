#!/usr/bin/env bash
# Find the replica's saturation throughput for a profile and print the 70% rate.
#
#   ./calibrate.sh [mixed|harsh]        # env: MODEL, CUDA_VISIBLE_DEVICES, TP, CONC (default 128), DURATION (default 180)
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
PROFILE=${1:-mixed}
CONC=${CONC:-128}
DURATION=${DURATION:-180}
PYTHON=${PYTHON:-python}
RUN="${RUNS_DIR:-$HERE/runs}/calibrate_${PROFILE}_$(date +%Y%m%d-%H%M%S)"
export FT_RUN_DIR="$RUN"
mkdir -p "$RUN/logs"
trap '"$PYTHON" "$HERE/stop.py" || true' EXIT INT TERM
"$HERE/launch.sh" || exit 1
"$PYTHON" "$HERE/recon.py" --json "$RUN/recon.json"
echo "[calibrate] closed loop, concurrency $CONC, ${DURATION}s, profile $PROFILE"
"$PYTHON" "$HERE/load.py" --closed-loop "$CONC" --duration "$DURATION" --profile "$PROFILE" --drain-timeout 300 | tee "$RUN/logs/load.out"
grep 'saturation' "$RUN/logs/load.out" | tail -1
echo "[calibrate] export RATE=<the number above> for run_case.sh / sweep_floor.sh"
