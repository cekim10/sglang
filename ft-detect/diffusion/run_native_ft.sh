#!/usr/bin/env bash
# Native NCCL fault tolerance vs issuance discipline (native_ft_probe.py), every case.
#   CUDA_VISIBLE_DEVICES=0,1 NCCL_P2P_DISABLE=1 ./run_native_ft.sh
# Env: ISSUES (none k1 k8 flood), RECOVERS (abort shrink_abort), NONBLOCKING (0 1)
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OUT_ROOT=$HERE/results/native_ft; mkdir -p "$OUT_ROOT"
for nb in ${NONBLOCKING:-0 1}; do
  for rec in ${RECOVERS:-abort shrink_abort}; do
    for iss in ${ISSUES:-none k1 k8 flood}; do
      name="${iss}_${rec}_nb${nb}"
      echo "=============== $name"
      TORCH_NCCL_USE_COMM_NONBLOCKING=$nb OUT_DIR="$OUT_ROOT/$name" MAX_S=180 "$HERE/run_pair.sh" native_ft_probe.py \
          --issue "$iss" --recover "$rec" | grep -E "init_done|deadline_miss|recovery_result|main_launch|new_work|device_sync|new_group|verdict|watchdog"
      sleep 2
    done
  done
done
python - "$OUT_ROOT" <<'PY'
import glob, json, os, sys
root = sys.argv[1]; rows = []; ver = None
for f in sorted(glob.glob(os.path.join(root, "*", "rank0.json"))):
    j = json.load(open(f)); c = j["case"]; ver = ver or f"torch {j.get('torch')}, NCCL {j.get('nccl')}, shrink_group {j.get('has_shrink_group')}"
    r = j.get("recovery", {}); nw = j.get("new_work_same_stream", {}); ds = j.get("device_sync", {}); ng = j.get("new_group_collective", {})
    rows.append(f"| {c['issue']} | {c['recover']} | {c['nonblocking']} | {('ok ' + str(r.get('ms')) + ' ms') if r.get('ok') else (r.get('err') or j.get('exit'))} | "
                f"{j.get('main_thread_unblocked')} | {nw.get('completed')} | {'ok' if ds.get('ok') else ds.get('err')} | "
                f"{('ok' if ng.get('ok') else ng.get('err')) if c['recover'] == 'shrink_abort' else '-'} | **{j.get('survivor_recoverable')}** | {j.get('exit')} |")
md = ["# Native NCCL fault tolerance vs issuance discipline (rank 0 = survivor)", "", ver or "", "",
      "| issued behind the stuck collective | recovery | nonblocking comm | recovery call | main thread unblocked | new work on same stream | torch.cuda.synchronize | collective on shrunk group | survivor recoverable | exit |",
      "|---|---|---|---|---|---|---|---|---|---|"] + rows + ["",
      "Gate: if survivor recoverable is True for k1/k8/flood with a native primitive, issuance discipline is unnecessary (KILL the issuance core). "
      "If only `none` is recoverable, the runtime's issuance decides whether NCCL's recovery stays usable (core survives)."]
open(os.path.join(root, "summary.md"), "w").write("\n".join(md) + "\n"); print("\n".join(md))
PY
