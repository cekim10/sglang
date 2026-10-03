#!/usr/bin/env bash
# Saturation throughput of the diffusion replica for a profile -> 70% rate.
#   ./calibrate_diff.sh [mixed|harsh]     # env: MODEL, CUDA_VISIBLE_DEVICES, NGPU, CONC (default 4), DURATION (default 420)
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/.." && pwd)
PROFILE=${1:-mixed}
CONC=${CONC:-4}
DURATION=${DURATION:-420}
PYTHON=${PYTHON:-python}
RUN="${RUNS_DIR:-$HERE/runs}/diff_calibrate_${PROFILE}_$(date +%Y%m%d-%H%M%S)"
export FT_RUN_DIR="$RUN" FT_API=diffusion
mkdir -p "$RUN/logs"
trap '"$PYTHON" "$ROOT/stop.py" || true' EXIT INT TERM
"$HERE/launch_diff.sh" || exit 1
"$PYTHON" "$HERE/recon_diff.py" --json "$RUN/recon.json"
echo "[calibrate] closed loop, concurrency $CONC, ${DURATION}s, profile $PROFILE (no batching: concurrency only fills the queue)"
"$PYTHON" "$HERE/load_diff.py" ${LOAD_ARGS:-} --closed-loop "$CONC" --duration "$DURATION" --profile "$PROFILE" --drain-timeout 600 | tee "$RUN/logs/load.out"
"$PYTHON" "$HERE/step_stats.py" "$RUN" --md "$RUN/step_stats.md" || true
grep 'saturation' "$RUN/logs/load.out" | tail -1
echo "[calibrate] export RATE=<the number above>; per-shape step times are in $RUN/step_stats.md"
