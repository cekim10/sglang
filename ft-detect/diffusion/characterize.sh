#!/usr/bin/env bash
# Workload-variability characterization (no fault injection): one model, a shape matrix,
# N sequential requests per shape, per-step and per-stage timings -> step_stats CSV/MD.
# This is the kill-test input: D_global vs D_w across image and video workloads.
#
#   SHAPES="832x480x33@30,832x480x81@30,1280x720x33@30,1280x720x81@30" ./characterize.sh wan22
#   SHAPES="256x256@9,512x512@9,1024x1024@9,1536x1536@9"                 ./characterize.sh zimage
#
# Positional: a short tag for the output files.
# Env: SHAPES (required; 'WxH[xF]@steps[:weight],...'), PER_SHAPE (default 5: 1 cold + 4 warm),
#      IDLE_PROBE (default 0; if N>0, sleep N s before the last request of each shape to sample
#      an after-idle cold start), plus launch_diff.sh's MODEL / CUDA_VISIBLE_DEVICES / NGPU /
#      PARALLEL_ARGS / EXTRA_ARGS / FT_PORT. Host quirks (NCCL_P2P_DISABLE, SGLANG_DIFFUSION_IPC_A2A)
#      are inherited from the shell.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/.." && pwd)
TAG=${1:?tag, e.g. wan22 or zimage}
: "${SHAPES:?set SHAPES}"
PER_SHAPE=${PER_SHAPE:-5}
IDLE_PROBE=${IDLE_PROBE:-0}
PYTHON=${PYTHON:-python}
RESULTS_DIR=${RESULTS_DIR:-$RESULTS_DIR}
RUN="${RUNS_DIR:-$HERE/runs}/diff_char_${TAG}_$(date +%Y%m%d-%H%M%S)"
export FT_RUN_DIR="$RUN" FT_API=diffusion
mkdir -p "$RUN/logs" "$RESULTS_DIR"
trap '"$PYTHON" "$ROOT/stop.py" || true' EXIT INT TERM

"$HERE/launch_diff.sh" || exit 1
"$PYTHON" "$HERE/recon_diff.py" --json "$RUN/recon.json" > "$RUN/logs/recon.out" 2>&1 || true
echo "{\"tag\":\"$TAG\",\"shapes\":\"$SHAPES\",\"per_shape\":$PER_SHAPE,\"idle_probe\":$IDLE_PROBE}" > "$RUN/characterize.json"

echo "[characterize] $TAG: $PER_SHAPE sequential requests per shape over: $SHAPES"
"$PYTHON" "$HERE/load_diff.py" ${LOAD_ARGS:-} --shapes "$SHAPES" --per-shape "$PER_SHAPE" --drain-timeout 1800 \
    --duration 100000 2>&1 | tee "$RUN/logs/load.out" | grep -v "^\[load\] t="
if [ "$IDLE_PROBE" -gt 0 ]; then
  echo "[characterize] idle probe: sleeping ${IDLE_PROBE}s then one request per shape"
  sleep "$IDLE_PROBE"
  "$PYTHON" "$HERE/load_diff.py" ${LOAD_ARGS:-} --shapes "$SHAPES" --per-shape 1 --tag idle --drain-timeout 1800 \
      --duration 100000 2>&1 | tee -a "$RUN/logs/load.out" | grep -v "^\[load\] t="
fi

"$PYTHON" "$HERE/step_stats.py" "$RUN" --md "$RESULTS_DIR/char_${TAG}.md" --csv "$RESULTS_DIR/char_${TAG}.csv"
echo "[characterize] wrote results/char_${TAG}.{md,csv}; merge all models with: python merge_floor.py results/char_*.csv"
