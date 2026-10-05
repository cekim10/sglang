#!/usr/bin/env bash
# Test 3: failure-contained distributed diffusion execution, end to end in SGLang Diffusion.
#
#   MODE=ours   ./run_contain.sh    SP=2 server with FT_CONTAIN=1. Request 1 = warm SP=2 reference
#                                   (same prompt/seed/shape). Request 2 = rank 1 SIGSTOPs itself at
#                                   step FAIL_STEP; rank 0 misses the DEADLINE_S, aborts, fails over to
#                                   SP=1 in-process and finishes the request. Request 3 = served by the
#                                   same process at SP=1 (is the process still a server?).
#   MODE=stock  ./run_contain.sh    same injection, no containment: the known no-detection path.
#                                   Bounded by STOCK_WAIT (default 700 s), then stopped by us.
#   MODE=sp1ref ./run_contain.sh    NGPU=1 server, one request: SP=1 reference output and latency.
#
# Env (plus launch_diff.sh's MODEL / CUDA_VISIBLE_DEVICES; export NCCL_P2P_DISABLE=1 and
# SGLANG_DIFFUSION_IPC_A2A=false on elves-01 as for every other diffusion run):
#   SHAPE        default 832x480x81       STEPS      default 9          SEED  default 1234
#   FAIL_STEP    default 4 (0-based step at which rank 1 freezes)
#   DEADLINE_S   default 5 (FT_CONTAIN_DEADLINE_S; the step time at this shape is ~0.9 s)
#   N            default 1 runs           RUNS_DIR / RESULTS_DIR as elsewhere
# Each run directory gets resp_ref.json / resp_fail.json / resp_after.json (+ outputs/*.mp4) and
# latents/<phase>_final_rank0.pt (final latents, for the numeric comparison in contain_report.py).
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/.." && pwd)
MODE=${MODE:-ours}
SHAPE=${SHAPE:-832x480x81}; STEPS=${STEPS:-9}; SEED=${SEED:-1234}
FAIL_STEP=${FAIL_STEP:-4}; DEADLINE_S=${DEADLINE_S:-5}
N=${N:-1}; STOCK_WAIT=${STOCK_WAIT:-700}
RUNS_DIR=${RUNS_DIR:-$HERE/runs}; RESULTS_DIR=${RESULTS_DIR:-$HERE/results}
PYTHON=${PYTHON:-python}
export FT_API=diffusion
mkdir -p "$RUNS_DIR" "$RESULTS_DIR"

one_req() {  # phase
  local phase=$1 out="$RUN/resp_$1.json"
  "$PYTHON" "$HERE/one_request.py" --shape "$SHAPE" --steps "$STEPS" --seed "$SEED" --out "$out" \
      --save "$RUN/outputs/$phase.bin" --timeout "${REQ_TIMEOUT:-1800}" | cut -c1-400
  mkdir -p "$RUN/latents"
  [ -e "$RUN/traj/final_rank0.pt" ] && mv "$RUN/traj/final_rank0.pt" "$RUN/latents/${phase}_final_rank0.pt"
  rm -f "$RUN/traj/final_rank1.pt"
}
cleanup_run() {
  "$PYTHON" "$ROOT/stop.py" || true; sleep 3
  [ -n "${PROBE_PID:-}" ] && kill -INT "$PROBE_PID" 2>/dev/null; wait 2>/dev/null; PROBE_PID=""
}
trap 'echo "[run_contain] interrupted"; cleanup_run; exit 130' INT TERM

for i in $(seq 1 "$N"); do
  RUN="$RUNS_DIR/contain_${MODE}_$(date +%Y%m%d-%H%M%S)"
  export FT_RUN_DIR="$RUN" FT_TRAJ_SAVE_DIR="$RUN/traj"
  mkdir -p "$RUN/logs" "$RUN/outputs"
  echo "=============== [Test 3] mode=$MODE shape=$SHAPE steps=$STEPS fail_step=$FAIL_STEP deadline=${DEADLINE_S}s run $i/$N -> $RUN"
  echo "{\"mode\":\"$MODE\",\"shape\":\"$SHAPE\",\"steps\":$STEPS,\"seed\":$SEED,\"fail_step\":$FAIL_STEP,\"deadline_s\":$DEADLINE_S}" > "$RUN/contain_env.json"
  case "$MODE" in
    ours)   export FT_CONTAIN=1 FT_CONTAIN_DEADLINE_S="$DEADLINE_S" FT_FAIL_STEP="$FAIL_STEP" FT_FAIL_RANK=1 FT_FAIL_REQ=2; L="$HERE/launch_diff.sh" ;;
    stock)  unset FT_CONTAIN; export FT_FAIL_STEP="$FAIL_STEP" FT_FAIL_RANK=1 FT_FAIL_REQ=2; L="$HERE/launch_diff.sh" ;;
    sp1ref) unset FT_CONTAIN FT_FAIL_STEP; L="env NGPU=1 PARALLEL_ARGS= CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES%%,*} $HERE/launch_diff.sh" ;;
    *) echo "unknown MODE=$MODE"; exit 2 ;;
  esac
  if ! $L; then echo "[run_contain] launch failed"; touch "$RUN/launch_failed"; "$PYTHON" "$ROOT/stop.py" || true; continue; fi
  "$PYTHON" "$ROOT/probe.py" --gen-timeout 60 > "$RUN/logs/probe.out" 2>&1 & PROBE_PID=$!

  if [ "$MODE" = sp1ref ]; then
    echo "[run_contain] warm request"; one_req warm
    echo "[run_contain] reference request (SP=1)"; one_req ref
  else
    echo "[run_contain] request 1: warm SP=2 reference"; one_req ref
    echo "[run_contain] request 2: rank 1 freezes at step $FAIL_STEP"
    if [ "$MODE" = ours ]; then
      one_req fail
      echo "[run_contain] request 3: same process, now SP=1"; one_req after
    else
      REQ_TIMEOUT="$STOCK_WAIT" one_req fail
      echo "[run_contain] stock: request outcome above after ${STOCK_WAIT}s bound; rank 1 state:"; grep -h '"who":"rank1"' "$RUN/probe.jsonl" | tail -1 | cut -c1-200
    fi
  fi
  cleanup_run
  "$PYTHON" "$HERE/contain_report.py" "$RUN" --md "$RUN/contain.md" > "$RUN/logs/report.out" 2>&1 && cat "$RUN/contain.md" || { echo "[report FAILED]"; tail -5 "$RUN/logs/report.out"; }
  sleep 5
done
echo "[run_contain] aggregate: python contain_report.py runs/contain_ours_* --sp1ref 'runs/contain_sp1ref_*' --md results/test3_contain.md --csv results/test3_contain.csv"
