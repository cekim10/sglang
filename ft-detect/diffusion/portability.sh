#!/usr/bin/env bash
# Phase 3b-1: trajectory portability across processes and SP degrees.
#
#   ./portability.sh <tag> <shape WxH[xF]> <steps> <k>
#   e.g. ./portability.sh wan_480p81 832x480x81 9 3      (MODEL=Wan-AI/Wan2.2-TI2V-5B-Diffusers)
#        ./portability.sh zimage_1024 1024x1024 9 3      (MODEL=Tongyi-MAI/Z-Image-Turbo)
#
# Runs the identical request (one_request.py: fixed prompt/seed) under five configurations,
# each in a fresh server:
#   sp2_save   SP=2, saves the boundary state after step k and the final latents
#   sp1_ref    SP=1 uninterrupted reference (final latents)
#   sp1_ref2   SP=1 uninterrupted again (run-to-run noise floor)
#   sp1_full   SP=1 resumed from sp2_save at step k+1 with the full solver history
#   sp1_lower  SP=1 resumed from sp2_save at step k+1 with the solver history reset (low-order restart)
# then trajectory_compare.py reports exactness / numerical error / restore and continuation timings.
# Env: MODEL, CUDA_VISIBLE_DEVICES, FT_PORT, plus host quirks (NCCL_P2P_DISABLE, SGLANG_DIFFUSION_IPC_A2A).
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/.." && pwd)
TAG=${1:?tag}; SHAPE=${2:?shape}; STEPS=${3:?steps}; K=${4:?k}
: "${MODEL:?set MODEL}"
PYTHON=${PYTHON:-python}
SEED=${SEED:-1234}
RESULTS_DIR=${RESULTS_DIR:-$HERE/results}
D="$RESULTS_DIR/traj_${TAG}"
RUNS_DIR=${RUNS_DIR:-$HERE/runs}
mkdir -p "$D"
echo "{\"tag\":\"$TAG\",\"model\":\"$MODEL\",\"shape\":\"$SHAPE\",\"steps\":$STEPS,\"k\":$K,\"seed\":$SEED}" > "$D/config.json"

run_cfg() {  # name ngpu parallel_args extra_env...
  local name=$1 ngpu=$2 par=$3; shift 3
  local RUN="$RUNS_DIR/traj_${TAG}_${name}_$(date +%Y%m%d-%H%M%S)"
  echo "=============== [$TAG] $name (ngpu=$ngpu) -> $RUN"
  mkdir -p "$D/$name"
  ( export FT_RUN_DIR="$RUN" NGPU="$ngpu" PARALLEL_ARGS="$par" FT_TRAJ_SAVE_DIR="$D/$name" "$@"
    "$HERE/launch_diff.sh" || exit 1
    "$PYTHON" "$HERE/one_request.py" --shape "$SHAPE" --steps "$STEPS" --seed "$SEED" --out "$D/$name/resp.json"
    rc=$?
    cp "$RUN/logs/steps_rank0.jsonl" "$D/$name/steps_rank0.jsonl" 2>/dev/null
    [ -e "$RUN/logs/steps_rank1.jsonl" ] && cp "$RUN/logs/steps_rank1.jsonl" "$D/$name/steps_rank1.jsonl"
    "$PYTHON" "$ROOT/stop.py" > /dev/null
    exit $rc )
  local rc=$?
  sleep 5
  return $rc
}

run_cfg sp2_save  2 "--sp-degree 2 --ulysses-degree 2" FT_TRAJ_SAVE_STEP="$K" || { echo "sp2_save failed"; exit 1; }
TRAJ="$D/sp2_save/traj_step${K}_rank0.pt"
[ -e "$TRAJ" ] || { echo "no trajectory file $TRAJ (check $D/sp2_save/steps_rank0.jsonl for traj records)"; exit 1; }
run_cfg sp1_ref   1 "" || echo "sp1_ref failed"
run_cfg sp1_ref2  1 "" || echo "sp1_ref2 failed"
run_cfg sp1_full  1 "" FT_TRAJ_RESUME="$TRAJ" FT_TRAJ_RESUME_STEP="$K" FT_TRAJ_MODE=full  || echo "sp1_full failed"
run_cfg sp1_lower 1 "" FT_TRAJ_RESUME="$TRAJ" FT_TRAJ_RESUME_STEP="$K" FT_TRAJ_MODE=lower || echo "sp1_lower failed"

"$PYTHON" "$HERE/trajectory_compare.py" "$D" --md "$D/compare.md"
echo "[portability] done: $D/compare.md"
