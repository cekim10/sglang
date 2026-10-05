#!/usr/bin/env bash
# Run abort_matrix_probe.py over the cases that separate "server layout" from "probe layout".
#   CUDA_VISIBLE_DEVICES=0,1 NCCL_P2P_DISABLE=1 ./run_abort_matrix.sh
# Env: CASES (default all below, "init:stuck:abort:gloo:fence"), TORCH_NCCL_ASYNC_ERROR_HANDLING as you like.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OUT_ROOT=$HERE/results/abort_matrix; mkdir -p "$OUT_ROOT"
CASES=${CASES:-"lazy:default:world:0:0 eager:default:world:0:0 eager:sub:world:0:0 eager:sub:world:1:0 eager:sub:stuck:1:0 eager:sub:world:1:1 eager:sub:stuck:1:1 eager:sub:none:1:1"}
for c in $CASES; do
  IFS=: read -r init stuck abort gloo fence <<< "$c"
  name="${init}_${stuck}_${abort}_gloo${gloo}_fence${fence}"
  echo "=============== $name"
  OUT_DIR="$OUT_ROOT/$name" MAX_S=120 "$HERE/run_pair.sh" abort_matrix_probe.py \
      --init "$init" --stuck "$stuck" --abort "$abort" --gloo "$gloo" --fence "$fence" | grep -E "abort_result|main_stream_after|device_sync_after|watchdog" 
  sleep 2
done
python - "$OUT_ROOT" <<'PY'
import json, sys, glob, os
root = sys.argv[1]; rows = []
for f in sorted(glob.glob(os.path.join(root, "*", "rank0.json"))):
    j = json.load(open(f)); c = j["case"]; ab = j.get("abort", {}); ms = j.get("main_stream_after", {}); ds = j.get("device_sync_after", {})
    rows.append(f"| {c['init']} | {c['stuck']} | {c['abort']} | {c['gloo']} | {c['fence']} | "
                f"{'ok ' + str(ab.get('ms')) + ' ms' if ab.get('ok') else (ab.get('err') or j.get('exit'))} | "
                f"{ms.get('completed')} | {'ok' if ds.get('ok') else ds.get('err')} | {j.get('exit')} |")
md = ["# abort matrix (rank 0)", "", "| init | stuck group | abort | gloo groups | fence first | abort | main-stream launch completes | torch.cuda.synchronize | exit |",
      "|---|---|---|---|---|---|---|---|---|"] + rows
open(os.path.join(root, "summary.md"), "w").write("\n".join(md) + "\n"); print("\n".join(md))
PY
