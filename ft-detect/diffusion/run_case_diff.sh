#!/usr/bin/env bash
# Phase 2 (diffusion) injection driver: N runs of one case, fresh server each time.
#
#   RATE=<req/s from calibrate_diff.sh> ./run_case_diff.sh <A|B> <target_rank> <n_runs>
#
# Env (besides launch_diff.sh's MODEL / CUDA_VISIBLE_DEVICES / NGPU / PARALLEL_ARGS / FT_PORT / DIST_TIMEOUT / RPC_TIMEOUT):
#   RATE              (required) open-loop rate
#   PROFILE           mixed | harsh | video      (default mixed)
#   RUNS_DIR          default ./runs
#   MIN_WARMUP        s of load before the steady check (default 180; image requests take ~10 s each)
#   STEADY_WAIT       default 900
#   STEADY_ARGS       default "--window 120 --min-reqs 6 --tol 0.5" (few, long requests)
#   MAX_DETECT_WAIT   default 4000: the diffusion stack has NO step watchdog; the only engine-side
#                     timeouts are the IPC all-to-all (10 s, 2-rank Ulysses only) and torch's
#                     dist_timeout (3600 s). Use the full wait on run 1 to learn which fires, then
#                     lower it (e.g. 900) for the remaining runs if the answer is deterministic.
#   POST_GRACE        default 300
#   INJECT_JITTER     default 0
#   FT_EXTERNAL_BENCH=1  after detection, run gpu_bench.py in a separate process on rank 0's GPU
#                     (B2 feasibility: is the GPU usable from another process during the hang?)
#   FT_ABORT_PROBE=1  Phase 3c: each rank aborts its torch process groups after FT_ABORT_DEADLINE_S
#                     (default 10) seconds without progress; containment.py writes the timeline.
#                     Use with MAX_DETECT_WAIT=120 POST_GRACE=60; TORCH_NCCL_ASYNC_ERROR_HANDLING=0 is
#                     the documented setting for _abort_process_group (try both).
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/.." && pwd)
CASE=${1:?case A|B}; RANK=${2:?target rank}; N=${3:?n_runs}
: "${RATE:?set RATE (req/s) from calibrate_diff.sh}"
PROFILE=${PROFILE:-mixed}
RUNS_DIR=${RUNS_DIR:-$HERE/runs}
RESULTS_DIR=${RESULTS_DIR:-$HERE/results}
MIN_WARMUP=${MIN_WARMUP:-180}
STEADY_WAIT=${STEADY_WAIT:-900}
STEADY_ARGS=${STEADY_ARGS:-"--window 120 --min-reqs 6 --tol 0.5"}
MAX_DETECT_WAIT=${MAX_DETECT_WAIT:-4000}
POST_GRACE=${POST_GRACE:-300}
INJECT_JITTER=${INJECT_JITTER:-0}
PYTHON=${PYTHON:-python}
LOAD_DURATION=$(( MIN_WARMUP + STEADY_WAIT + INJECT_JITTER + MAX_DETECT_WAIT + POST_GRACE + 120 ))
export FT_API=diffusion
mkdir -p "$RUNS_DIR" "$RESULTS_DIR"

DETECT_RE='"pattern":"(watchdog_fire|scheduler_exception|subprocess_crashed|sigquit|kill_tree|nccl_timeout|nccl_error|abort|torch_dist_error|cuda_error|scheduler_terminated|diff_recv_error|diff_exec_error|diff_max_errors|diff_worker_dead|diff_worker_shutdown|diff_ipc_a2a_timeout|ft_abort_miss|ft_abort_done|ft_contain_miss|ft_contain_done)"|"who":"(http_server|rank0)","pid":[0-9]+,"state":"(dead|zombie)"'

cleanup_run() {
  "$PYTHON" "$ROOT/stop.py" || true
  sleep 3
  [ -n "${LOAD_PID:-}" ] && kill -INT "$LOAD_PID" 2>/dev/null
  [ -n "${PROBE_PID:-}" ] && kill -INT "$PROBE_PID" 2>/dev/null
  wait 2>/dev/null
  LOAD_PID=""; PROBE_PID=""
}
trap 'echo "[run_case_diff] interrupted"; cleanup_run; exit 130' INT TERM

for i in $(seq 1 "$N"); do
  RUN="$RUNS_DIR/diff_${CASE}_r${RANK}_${i}_$(date +%Y%m%d-%H%M%S)"
  export FT_RUN_DIR="$RUN"
  mkdir -p "$RUN/logs"
  echo "=============== [diffusion] case $CASE rank $RANK run $i/$N -> $RUN"
  if ! "$HERE/launch_diff.sh"; then
    echo "[run_case_diff] launch failed"; touch "$RUN/launch_failed"; "$PYTHON" "$ROOT/stop.py" || true; continue
  fi
  "$PYTHON" "$HERE/recon_diff.py" --json "$RUN/recon.json" > "$RUN/logs/recon.out" 2>&1 || true

  "$PYTHON" "$ROOT/probe.py" --gen-timeout 60 > "$RUN/logs/probe.out" 2>&1 & PROBE_PID=$!
  "$PYTHON" "$HERE/load_diff.py" ${LOAD_ARGS:-} --rate "$RATE" --duration "$LOAD_DURATION" --profile "$PROFILE" --tag "$i" \
      > "$RUN/logs/load.out" 2>&1 & LOAD_PID=$!
  echo "[run_case_diff] probe $PROBE_PID load $LOAD_PID; warming up ${MIN_WARMUP}s"
  sleep "$MIN_WARMUP"
  # shellcheck disable=SC2086
  if ! "$PYTHON" "$ROOT/steady.py" $STEADY_ARGS --wait "$STEADY_WAIT"; then
    echo "[run_case_diff] WARNING: steady state not reached; injecting anyway"; touch "$RUN/not_steady"
  fi
  if [ "$INJECT_JITTER" -gt 0 ]; then
    j=$(( RANDOM % (INJECT_JITTER + 1) )); echo "[run_case_diff] jitter ${j}s"; echo "$j" > "$RUN/inject_jitter_s"; sleep "$j"
  fi

  n_before=$(wc -l < "$RUN/probe.jsonl" 2>/dev/null || echo 0)
  "$PYTHON" "$ROOT/inject.py" --case "$CASE" --rank "$RANK" --run-id "$(basename "$RUN")" || { cleanup_run; continue; }

  t0=$(date +%s); detected=0
  while [ $(( $(date +%s) - t0 )) -lt "$MAX_DETECT_WAIT" ]; do
    if tail -n +"$(( n_before + 1 ))" "$RUN/probe.jsonl" 2>/dev/null | grep -Eq "$DETECT_RE"; then
      detected=1; echo "[run_case_diff] engine-side detection after $(( $(date +%s) - t0 )) s"; break
    fi
    sleep 2
  done
  [ "$detected" = 1 ] || { echo "[run_case_diff] NO engine-side detection within ${MAX_DETECT_WAIT}s"; touch "$RUN/no_engine_detect"; }
  if [ "${FT_EXTERNAL_BENCH:-0}" = "1" ]; then
    # B2 feasibility: benchmark the surviving rank's GPU from a separate process during the hang
    GPU0=$(echo "$CUDA_VISIBLE_DEVICES" | cut -d, -f1)
    echo "[run_case_diff] external GPU bench on device $GPU0 during the hang"
    CUDA_VISIBLE_DEVICES="$GPU0" "$PYTHON" "$HERE/gpu_bench.py" --out "$RUN/external_bench_during_hang.json" | cut -c1-300
  fi

  SERVER_PID=$("$PYTHON" -c "import json;print(json.load(open('$RUN/pids.json'))['http_server'])" 2>/dev/null || echo 0)
  t1=$(date +%s); torn=0
  while [ $(( $(date +%s) - t1 )) -lt "$POST_GRACE" ]; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null || [ "$(ps -o stat= -p "$SERVER_PID" 2>/dev/null | cut -c1)" = "Z" ]; then
      torn=1; echo "[run_case_diff] replica torn down by the stack $(( $(date +%s) - t0 )) s after injection"; sleep 5; break
    fi
    sleep 2
  done
  [ "$torn" = 1 ] || { echo "[run_case_diff] stack did not tear the replica down within ${POST_GRACE}s; stopping it ourselves"; touch "$RUN/no_self_teardown"; }

  cleanup_run
  "$PYTHON" "$ROOT/analyze.py" "$RUN" --gap 60 --csv "$RESULTS_DIR/case${CASE}.csv" --md "$RESULTS_DIR/case${CASE}.md" > "$RUN/logs/analyze.out" 2>&1 || { echo "[analyze FAILED] see $RUN/logs/analyze.out"; tail -5 "$RUN/logs/analyze.out"; }
  if [ "${FT_ABORT_PROBE:-0}" = "1" ]; then
    "$PYTHON" "$HERE/containment.py" "$RUN" --md "$RUN/containment.md" --csv "$RUN/containment.csv" > "$RUN/logs/containment.out" 2>&1 \
      && "$PYTHON" "$HERE/containment.py" "$RUNS_DIR"/diff_${CASE}_r${RANK}_* --md "$RESULTS_DIR/containment_${CASE}.md" --csv "$RESULTS_DIR/containment_${CASE}.csv" > /dev/null 2>&1 \
      || { echo "[containment FAILED] see $RUN/logs/containment.out"; tail -5 "$RUN/logs/containment.out"; }
  fi
  sleep 10
done
echo "[run_case_diff] done: $RESULTS_DIR/case${CASE}.md"
