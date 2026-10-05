#!/usr/bin/env bash
# Test 3: failure-contained distributed diffusion execution, end to end in SGLang Diffusion.
#
#   MODE=ours   ./run_contain.sh    SP=2 server with FT_CONTAIN=1. Requests 1-2 = warm-up + SP=2
#                                   reference (same prompt/seed/shape). Request 3 = rank 1 SIGSTOPs
#                                   itself at step FAIL_STEP; rank 0 misses the DEADLINE_S, aborts,
#                                   fails over to SP=1 in-process and finishes the request. Request 4 =
#                                   served by the same process at SP=1 (is the process still a server?).
#   MODE=stock  ./run_contain.sh    same injection, no containment: the known no-detection path.
#                                   Bounded by STOCK_WAIT (default 700 s), then stopped by us.
#   MODE=sp1ref ./run_contain.sh    NGPU=1 server, one request: SP=1 reference output and latency.
#
# Env (plus launch_diff.sh's MODEL / CUDA_VISIBLE_DEVICES; export NCCL_P2P_DISABLE=1 and
# SGLANG_DIFFUSION_IPC_A2A=false on elves-01 as for every other diffusion run):
#   SHAPE        default 832x480x81       STEPS      default 9          SEED  default 1234
#   FAIL_STEP    default 4 (0-based step at which rank 1 freezes)
#   DEADLINE_S   default 5 (FT_CONTAIN_DEADLINE_S; the step time at this shape is ~0.7 s)
#   ABORT        stuck (default) | world | none   which communicators to abort (see abort_matrix_probe.py)
#   FENCE_FIRST  default 1: kill the peer before the abort   ABORT_TIMEOUT_S default 10 (bounded abort)
#   TORCH_NCCL_ASYNC_ERROR_HANDLING defaults to 0 in MODE=ours (torch's documented setting for abort;
#                otherwise the NCCL watchdog tears the surviving rank down at 600 s if an abort stalls)
#   N            default 1 runs           RUNS_DIR / RESULTS_DIR as elsewhere
# Each run directory gets resp_{warm,ref,fail,after}.json (+ outputs/*.bin, rank_states.jsonl) and
# latents/<phase>_final_rank0.pt (final latents, for the numeric comparison in contain_report.py).
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/.." && pwd)
MODE=${MODE:-ours}
SHAPE=${SHAPE:-832x480x81}; STEPS=${STEPS:-9}; SEED=${SEED:-1234}
FAIL_STEP=${FAIL_STEP:-4}; DEADLINE_S=${DEADLINE_S:-5}
ABORT=${ABORT:-stuck}; FENCE_FIRST=${FENCE_FIRST:-1}; ABORT_TIMEOUT_S=${ABORT_TIMEOUT_S:-10}
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
  wait 2>/dev/null
}
trap 'echo "[run_contain] interrupted"; cleanup_run; exit 130' INT TERM

for i in $(seq 1 "$N"); do
  RUN="$RUNS_DIR/contain_${MODE}_$(date +%Y%m%d-%H%M%S)"
  export FT_RUN_DIR="$RUN" FT_TRAJ_SAVE_DIR="$RUN/traj"
  mkdir -p "$RUN/logs" "$RUN/outputs"
  echo "=============== [Test 3] mode=$MODE shape=$SHAPE steps=$STEPS fail_step=$FAIL_STEP deadline=${DEADLINE_S}s run $i/$N -> $RUN"
  echo "{\"mode\":\"$MODE\",\"shape\":\"$SHAPE\",\"steps\":$STEPS,\"seed\":$SEED,\"fail_step\":$FAIL_STEP,\"deadline_s\":$DEADLINE_S,\"abort\":\"$ABORT\",\"fence_first\":$FENCE_FIRST,\"abort_timeout_s\":$ABORT_TIMEOUT_S,\"nccl_async_error_handling\":\"${TORCH_NCCL_ASYNC_ERROR_HANDLING:-0}\"}" > "$RUN/contain_env.json"
  case "$MODE" in
    ours)   export FT_CONTAIN=1 FT_CONTAIN_DEADLINE_S="$DEADLINE_S" FT_FAIL_STEP="$FAIL_STEP" FT_FAIL_RANK=1 FT_FAIL_REQ=3 FT_FAIL_SHAPE="$SHAPE"
            export FT_CONTAIN_ABORT="$ABORT" FT_CONTAIN_FENCE_FIRST="$FENCE_FIRST" FT_CONTAIN_ABORT_TIMEOUT_S="$ABORT_TIMEOUT_S"
            export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-0}"; L="$HERE/launch_diff.sh" ;;
    stock)  unset FT_CONTAIN; export FT_FAIL_STEP="$FAIL_STEP" FT_FAIL_RANK=1 FT_FAIL_REQ=3 FT_FAIL_SHAPE="$SHAPE"; L="$HERE/launch_diff.sh" ;;
    sp1ref) unset FT_CONTAIN FT_FAIL_STEP; L="env NGPU=1 PARALLEL_ARGS= CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES%%,*} $HERE/launch_diff.sh" ;;
    *) echo "unknown MODE=$MODE"; exit 2 ;;
  esac
  if ! $L; then echo "[run_contain] launch failed"; touch "$RUN/launch_failed"; "$PYTHON" "$ROOT/stop.py" || true; continue; fi
  # no probe.py here: its periodic generation requests would interleave with ours and advance the
  # injection's request count; rank states are sampled with ps instead
  rank_states() { "$PYTHON" - "$RUN" "$1" <<'PY'
import json, subprocess, sys, time
run, tag = sys.argv[1], sys.argv[2]
pids = json.load(open(run + "/pids.json"))["ranks"]
st = {r: (subprocess.run(["ps", "-o", "stat=", "-p", str(p)], capture_output=True, text=True).stdout.strip() or "dead") for r, p in pids.items()}
open(run + "/rank_states.jsonl", "a").write(json.dumps({"tag": tag, "t": time.time(), "states": st}) + "\n")
print(f"[run_contain] rank states {tag}: {st}")
PY
  }
  if [ "$MODE" = sp1ref ]; then
    echo "[run_contain] warm request"; one_req warm
    echo "[run_contain] reference request (SP=1)"; one_req ref
  else
    echo "[run_contain] request 1: warm-up of this shape (SP=2)"; one_req warm
    echo "[run_contain] request 2: SP=2 reference"; one_req ref
    rank_states before_fail
    echo "[run_contain] request 3: rank 1 freezes at step $FAIL_STEP"
    if [ "$MODE" = ours ]; then
      REQ_TIMEOUT="${REQ_TIMEOUT:-300}" one_req fail; rank_states after_fail
      echo "[run_contain] request 4: same process, now SP=1"; one_req after; rank_states after_after
    else
      REQ_TIMEOUT="$STOCK_WAIT" one_req fail; rank_states after_fail
    fi
  fi
  cleanup_run
  "$PYTHON" "$HERE/contain_report.py" "$RUN" --md "$RUN/contain.md" > "$RUN/logs/report.out" 2>&1 && cat "$RUN/contain.md" || { echo "[report FAILED]"; tail -5 "$RUN/logs/report.out"; }
  sleep 5
done
echo "[run_contain] aggregate: python contain_report.py runs/contain_ours_* --sp1ref 'runs/contain_sp1ref_*' --md results/test3_contain.md --csv results/test3_contain.csv"
