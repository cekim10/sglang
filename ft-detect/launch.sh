#!/usr/bin/env bash
# Start one SGLang replica (--tp-size $TP) through launch_wrapped.py and block
# until /health returns 200 and run/pids.json exists.
#
# Env:
#   MODEL             (required) HF id or local path
#   CUDA_VISIBLE_DEVICES (required) the GPUs allocated to you; refuses to run if unset
#   TP                default 4
#   FT_PORT           default 30000
#   FT_RUN_DIR        default ./run   (per-run directory; must not already hold pids.json)
#   WATCHDOG_TIMEOUT  seconds; empty = stock default
#   DIST_TIMEOUT      seconds; empty = stock default (torch default)
#   EXTRA_ARGS        extra CLI flags passed verbatim
#   READY_TIMEOUT     seconds to wait for readiness, default 1800
#   PYTHON            interpreter, default python
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
: "${MODEL:?set MODEL=<hf id or local path>}"
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to the GPUs allocated to you}"
TP=${TP:-4}
PORT=${FT_PORT:-30000}
RUN_DIR=${FT_RUN_DIR:-$HERE/run}
READY_TIMEOUT=${READY_TIMEOUT:-1800}
PYTHON=${PYTHON:-python}
export FT_RUN_DIR="$RUN_DIR" FT_PORT="$PORT"

if [ -e "$RUN_DIR/pids.json" ]; then
  echo "refusing to start: $RUN_DIR/pids.json exists (stale run?). Use a fresh FT_RUN_DIR or run stop.py first." >&2
  exit 2
fi
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/pids"

args=(--model-path "$MODEL" --tp-size "$TP" --port "$PORT" --host 127.0.0.1)
[ -n "${WATCHDOG_TIMEOUT:-}" ] && args+=(--watchdog-timeout "$WATCHDOG_TIMEOUT")
[ -n "${DIST_TIMEOUT:-}" ] && args+=(--dist-timeout "$DIST_TIMEOUT")
# shellcheck disable=SC2206
extra=(${EXTRA_ARGS:-})

echo "[launch] GPUs=$CUDA_VISIBLE_DEVICES model=$MODEL tp=$TP port=$PORT run_dir=$RUN_DIR"
echo "[launch] args: ${args[*]} ${extra[*]:-}"
{
  echo "{\"model\":\"$MODEL\",\"tp\":$TP,\"port\":$PORT,\"cuda_visible_devices\":\"$CUDA_VISIBLE_DEVICES\","
  echo " \"watchdog_timeout\":\"${WATCHDOG_TIMEOUT:-default}\",\"dist_timeout\":\"${DIST_TIMEOUT:-default}\",\"extra_args\":\"${EXTRA_ARGS:-}\"}"
} > "$RUN_DIR/launch_env.json"

setsid nohup "$PYTHON" "$HERE/launch_wrapped.py" "${args[@]}" "${extra[@]}" \
  > "$RUN_DIR/logs/server.log" 2>&1 < /dev/null &
SERVER_PID=$!
echo "$SERVER_PID" > "$RUN_DIR/pids/server.pid"
echo "[launch] server pid (shell view) $SERVER_PID; log $RUN_DIR/logs/server.log"

t0=$(date +%s)
while true; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[launch] server process exited during startup; tail of log:" >&2
    tail -n 40 "$RUN_DIR/logs/server.log" >&2
    exit 1
  fi
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 30 "http://127.0.0.1:$PORT/health" || true)
  if [ "$code" = "200" ] && [ -e "$RUN_DIR/pids.json" ]; then
    break
  fi
  if [ $(( $(date +%s) - t0 )) -ge "$READY_TIMEOUT" ]; then
    echo "[launch] timed out waiting for readiness (last /health=$code)" >&2
    exit 1
  fi
  sleep 2
done
echo "[launch] ready after $(( $(date +%s) - t0 )) s"
cat "$RUN_DIR/pids.json"
"$PYTHON" "$HERE/recon.py" --tree || true
