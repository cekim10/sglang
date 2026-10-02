#!/usr/bin/env bash
# Start one SGLang Diffusion replica (--num-gpus $NGPU) through launch_wrapped_diff.py and
# block until /health returns 200 (warmup done) and run/pids.json exists.
#
# Env:
#   MODEL                 (required) HF id or local path, e.g. black-forest-labs/FLUX.1-dev
#   CUDA_VISIBLE_DEVICES  (required) your GPUs
#   NGPU                  default 2  (-> --num-gpus)
#   PARALLEL_ARGS         default "--sp-degree 2 --ulysses-degree 2" (per-step collective between ranks)
#   FT_PORT               default 30000
#   FT_RUN_DIR            default ./run
#   DIST_TIMEOUT          s; empty = stock default (3600)
#   RPC_TIMEOUT           s; empty = stock default (None = no timeout)   (-> --scheduler-rpc-timeout)
#   EXTRA_ARGS            extra CLI flags verbatim
#   READY_TIMEOUT         default 2400 (model load + warmup)
#   PYTHON                default python
# Host quirks are NOT set here; export them yourself (elves-01 needs NCCL_P2P_DISABLE=1).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
: "${MODEL:?set MODEL}"
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to the GPUs allocated to you}"
NGPU=${NGPU:-2}
PARALLEL_ARGS=${PARALLEL_ARGS-"--sp-degree 2 --ulysses-degree 2"}
PORT=${FT_PORT:-30000}
RUN_DIR=${FT_RUN_DIR:-$HERE/run}
READY_TIMEOUT=${READY_TIMEOUT:-2400}
PYTHON=${PYTHON:-python}
export FT_RUN_DIR="$RUN_DIR" FT_PORT="$PORT" FT_API=diffusion

if [ -e "$RUN_DIR/pids.json" ]; then
  echo "refusing to start: $RUN_DIR/pids.json exists (stale run?)" >&2; exit 2
fi
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/pids"

args=(--model-path "$MODEL" --num-gpus "$NGPU" --port "$PORT" --host 127.0.0.1)
# shellcheck disable=SC2206
par=($PARALLEL_ARGS)
[ -n "${DIST_TIMEOUT:-}" ] && args+=(--dist-timeout "$DIST_TIMEOUT")
[ -n "${RPC_TIMEOUT:-}" ] && args+=(--scheduler-rpc-timeout "$RPC_TIMEOUT")
# shellcheck disable=SC2206
extra=(${EXTRA_ARGS:-})

echo "[launch] GPUs=$CUDA_VISIBLE_DEVICES model=$MODEL num_gpus=$NGPU port=$PORT run_dir=$RUN_DIR"
echo "[launch] args: ${args[*]} ${par[*]:-} ${extra[*]:-}"
{
  echo "{\"model\":\"$MODEL\",\"tp\":$NGPU,\"port\":$PORT,\"cuda_visible_devices\":\"$CUDA_VISIBLE_DEVICES\",\"stack\":\"sglang-diffusion\","
  echo " \"watchdog_timeout\":\"absent\",\"dist_timeout\":\"${DIST_TIMEOUT:-default}\",\"rpc_timeout\":\"${RPC_TIMEOUT:-default}\","
  echo " \"parallel_args\":\"${PARALLEL_ARGS}\",\"extra_args\":\"${EXTRA_ARGS:-}\",\"ipc_a2a\":\"${SGLANG_DIFFUSION_IPC_A2A:-default}\",\"nccl_p2p_disable\":\"${NCCL_P2P_DISABLE:-}\"}"
} > "$RUN_DIR/launch_env.json"

setsid nohup "$PYTHON" "$HERE/launch_wrapped_diff.py" "${args[@]}" "${par[@]}" "${extra[@]}" \
  > "$RUN_DIR/logs/server.log" 2>&1 < /dev/null &
SERVER_PID=$!
echo "$SERVER_PID" > "$RUN_DIR/pids/server.pid"
echo "[launch] server pid $SERVER_PID; log $RUN_DIR/logs/server.log"

t0=$(date +%s)
while true; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[launch] server process exited during startup; tail of log:" >&2
    tail -n 40 "$RUN_DIR/logs/server.log" >&2
    "$PYTHON" "$HERE/../stop.py" >/dev/null 2>&1 || true
    exit 1
  fi
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "http://127.0.0.1:$PORT/health" || true)
  if [ "$code" = "200" ] && [ -e "$RUN_DIR/pids.json" ]; then break; fi
  if [ $(( $(date +%s) - t0 )) -ge "$READY_TIMEOUT" ]; then
    echo "[launch] timed out waiting for readiness (last /health=$code); tail of log:" >&2
    tail -n 40 "$RUN_DIR/logs/server.log" >&2
    "$PYTHON" "$HERE/../stop.py" || true
    exit 1
  fi
  sleep 2
done
echo "[launch] ready after $(( $(date +%s) - t0 )) s"
cat "$RUN_DIR/pids.json"
"$PYTHON" "$HERE/recon_diff.py" --tree || true
