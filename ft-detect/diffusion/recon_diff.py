"""Reconnaissance of the *installed* SGLang Diffusion runtime (sglang.multimodal_gen).

    python recon_diff.py            # defaults with file:line -> run/recon.json + stdout table
    python recon_diff.py --tree     # process tree of the running replica from run/pids.json

Phase 2 question for the diffusion stack: which detectors exist at all. The
Phase 1 code read found no step watchdog, a readiness-only /health, no RPC
timeout and a 1 h dist timeout; this script re-derives each from the installed
source so the report quotes the code that ran.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import run_dir  # noqa: E402
from recon import _find, _git_commit, _grep, tree  # noqa: E402


def recon():
    out = {"python": sys.version.split()[0], "stack": "sglang-diffusion"}
    try:
        import sglang

        root = Path(sglang.__file__).resolve().parent
        out["sglang_version"] = getattr(sglang, "__version__", "?")
        out["sglang_path"] = str(root)
        out["sglang_git_commit"], _ = _git_commit(root)
    except Exception as e:
        out["sglang_error"] = repr(e)
        return out
    try:
        import torch

        out["torch_version"] = torch.__version__
        from torch.distributed import constants as c

        out["torch_default_pg_nccl_timeout_s"] = c.default_pg_nccl_timeout.total_seconds()
    except Exception as e:
        out["torch_error"] = repr(e)
    out["env"] = {k: os.environ.get(k) for k in [
        "SGLANG_DIFFUSION_IPC_A2A", "SGLANG_DIFFUSION_IPC_A2A_TIMEOUT_MS", "NCCL_P2P_DISABLE",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING", "CUDA_VISIBLE_DEVICES"]}

    mg = root / "multimodal_gen"
    if not mg.exists():
        out["error"] = "sglang.multimodal_gen not installed (pip install 'sglang[diffusion]')"
        return out
    rt = mg / "runtime"
    f = {}

    def rec(name, path, pattern, note=""):
        val, line = _grep(path, pattern)
        f[name] = {"value": val, "file": str(path.relative_to(root.parent)) if path else None, "line": line, "note": note}

    sa = _find(rt, ["server_args/server_args.py"])
    rec("dist_timeout_default", sa, r"\bdist_timeout:\s*int\s*\|\s*None\s*=\s*(\d+)", "torch.distributed PG timeout; the only timeout on the NCCL fallback path")
    rec("scheduler_rpc_timeout_default", sa, r"\bscheduler_rpc_timeout:\s*int\s*\|\s*None\s*=\s*(\w+)", "HTTP -> rank-0 scheduler ZMQ RCVTIMEO; None = wait forever")
    rec("watchdog_timeout_arg", sa, r"(watchdog_timeout)", "ABSENT in diffusion ServerArgs if NOT FOUND: no step watchdog exists")
    rec("num_gpus_default", sa, r"\bnum_gpus:\s*int\s*=\s*(\d+)")

    hs = _find(rt, ["entrypoints/http_server.py"])
    rec("health_handler", hs, r'@health_router\.get\("/health"\)\s*\n\s*async def health\(request: Request\):\s*\n\s*"""([^"]*)"""', "readiness flag only")
    rec("health_body", hs, r'(if not request\.app\.state\.server_warmup_done\.is_set\(\):\s*\n\s*return Response\(status_code=503\)\s*\n\s*return \{"status": "ok"\})', "200 forever once warm; never consults the engine")
    rec("health_generate_body", hs, r'async def health_generate\(request: Request\):\s*\n\s*"""([^"]*)"""', "no generation issued")

    sc = _find(rt, ["scheduler_client.py"])
    rec("rpc_rcvtimeo_set", sc, r"(socket\.setsockopt\(zmq\.RCVTIMEO, timeout_ms\))", "only when a timeout is configured")
    rec("rpc_ping_timeout_ms", sc, r"ping_socket\.setsockopt\(zmq\.RCVTIMEO,\s*(\d+)\)", "liveness ping used by the client, not by /health")

    sch = _find(rt, ["managers/scheduler.py"])
    rec("max_consecutive_recv_errors", sch, r"self\._max_consecutive_errors\s*=\s*(\d+)", "rank-0 loop exits after this many recv errors")
    rec("recv_error_msg", sch, r'(f?"Error receiving requests in scheduler event loop)')
    rec("exec_error_msg", sch, r'(f?"Error executing request in scheduler event loop)')
    rec("slave_result_recv", sch, r"(results\.append\(pipe\.recv\(\)\))", "rank 0 blocks here on a hung slave; no timeout")

    gw = _find(rt, ["managers/gpu_worker.py"])
    rec("pdeathsig", gw, r"(kill_itself_when_parent_died\(\))", "workers get SIGKILL when the HTTP process dies")
    rec("worker_shutdown_msg", gw, r'(f?"Worker \{rank\}: Shutdown complete\.")')

    ls = _find(rt, ["launch_server.py"])
    rec("worker_dead_msg", ls, r'(f?"Rank \{rank_offset \+ i\} scheduler is dead)', "only checked during startup")
    rec("worker_proc_name", ls, r'name=(f"sglang-diffusionWorker-\{rank\}")')

    envf = _find(mg, ["envs.py"])
    rec("ipc_a2a_default", envf, r'"SGLANG_DIFFUSION_IPC_A2A":\s*_lazy_bool\("SGLANG_DIFFUSION_IPC_A2A",\s*"(\w+)"\)', "2-rank Ulysses all-to-all over CUDA IPC")
    rec("ipc_a2a_timeout_ms", envf, r'"SGLANG_DIFFUSION_IPC_A2A_TIMEOUT_MS",\s*([\d.]+)', "the ONLY sub-minute timeout on the per-step collective path")

    bdc = _find(rt, ["distributed/device_communicators/base_device_communicator.py"])
    rec("a2a_torch_fallback", bdc, r"(dist\.all_to_all_single\()", "when IPC a2a is off/unavailable: torch PG op, covered by dist_timeout")

    dn = _find(rt, ["pipelines_core/stages/denoising.py"])
    rec("denoise_step_def", dn, r"(def _run_denoising_step\()", "wrapped by launch_wrapped_diff.py for per-step timestamps")
    rec("denoise_loop", dn, r"(for step_index, t_host in enumerate\(timesteps_cpu\))")

    out["findings"] = f
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", action="store_true")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    if a.tree:
        return tree()
    out = recon()
    path = Path(a.json) if a.json else run_dir() / "recon.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(f"sglang {out.get('sglang_version')} (diffusion) @ {out.get('sglang_path')} torch {out.get('torch_version')}")
    print("| finding | value | file:line |\n|---|---|---|")
    for k, v in out.get("findings", {}).items():
        loc = f"{v['file']}:{v['line']}" if v.get("line") else "NOT FOUND"
        val = (v["value"] or "")[:70].replace("\n", " ")
        print(f"| {k} | `{val}` | {loc} |")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
