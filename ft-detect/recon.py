"""Reconnaissance of the *installed* SGLang/torch on this host.

    python recon.py            # version, commit, timeout defaults with file:line -> run/recon.json + stdout table
    python recon.py --gpus     # which GPUs are free (no compute processes)
    python recon.py --tree     # process tree of the running replica from run/pids.json

The defaults are read from the installed source with regexes, so the numbers in
REPORT.md come from the code that actually ran, not from this checkout.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_pids, run_dir  # noqa: E402


def _find(root: Path, rel_candidates):
    for rel in rel_candidates:
        p = root / rel
        if p.exists():
            return p
    return None


def _grep(path: Path, pattern: str, flags=re.M | re.S, group: int = 1):
    """Return (value, line_no) of the first regex match in file, or (None, None)."""
    if path is None or not path.exists():
        return None, None
    text = path.read_text(errors="replace")
    m = re.search(pattern, text, flags)
    if not m:
        return None, None
    line = text.count("\n", 0, m.start(group) if m.lastindex else m.start()) + 1
    return (m.group(group) if m.lastindex else m.group(0)).strip(), line


def _git_commit(pkg_dir: Path):
    for parent in [pkg_dir, *pkg_dir.parents]:
        if (parent / ".git").exists():
            try:
                h = subprocess.check_output(["git", "-C", str(parent), "rev-parse", "HEAD"], text=True).strip()
                dirty = subprocess.call(["git", "-C", str(parent), "diff", "--quiet"]) != 0
                return h + (" (dirty)" if dirty else ""), str(parent)
            except Exception:
                return None, str(parent)
    return None, None


def recon():
    out = {"python": sys.version.split()[0]}
    try:
        import sglang

        root = Path(sglang.__file__).resolve().parent
        out["sglang_version"] = getattr(sglang, "__version__", "?")
        out["sglang_path"] = str(root)
        out["sglang_git_commit"], out["sglang_git_root"] = _git_commit(root)
    except Exception as e:
        out["sglang_error"] = repr(e)
        root = None

    try:
        import torch

        out["torch_version"] = torch.__version__
        try:
            out["nccl_version"] = ".".join(map(str, torch.cuda.nccl.version()))
        except Exception:
            out["nccl_version"] = None
        try:
            from torch.distributed import constants as c

            out["torch_default_pg_timeout_s"] = c.default_pg_timeout.total_seconds()
            out["torch_default_pg_nccl_timeout_s"] = (
                c.default_pg_nccl_timeout.total_seconds() if getattr(c, "default_pg_nccl_timeout", None) else None
            )
        except Exception as e:
            out["torch_constants_error"] = repr(e)
    except Exception as e:
        out["torch_error"] = repr(e)

    out["env"] = {
        k: os.environ.get(k)
        for k in [
            "TORCH_NCCL_ASYNC_ERROR_HANDLING",
            "NCCL_ASYNC_ERROR_HANDLING",
            "TORCH_NCCL_BLOCKING_WAIT",
            "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC",
            "TORCH_NCCL_ENABLE_MONITORING",
            "SGLANG_HEALTH_CHECK_TIMEOUT",
            "SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION",
            "SGLANG_KILLPG_ON_SCHEDULER_EXCEPTION",
            "CUDA_VISIBLE_DEVICES",
        ]
    }

    if root is None:
        return out

    srt = root / "srt"
    findings = {}

    def rec(name, path, pattern, note=""):
        val, line = _grep(path, pattern)
        findings[name] = {
            "value": val,
            "file": str(path.relative_to(root.parent)) if path else None,
            "line": line,
            "note": note,
        }

    device_fields = _find(srt, ["arg_groups/fields/device.py", "server_args.py"])
    rec("watchdog_timeout_default", device_fields, r"\bwatchdog_timeout:\s*(?:A\[.*?\]|float|Optional\[float\])\s*=\s*([\d.]+)")
    rec("soft_watchdog_timeout_default", device_fields, r"\bsoft_watchdog_timeout:\s*(?:A\[.*?\]|Optional\[float\])\s*=\s*(\w+)")
    parallel_fields = _find(srt, ["arg_groups/fields/parallel.py", "server_args.py"])
    rec("dist_timeout_default", parallel_fields, r"\bdist_timeout:\s*(?:A\[.*?\]|Optional\[int\])\s*=\s*(\w+)", "None -> torch default")

    wd = _find(srt, ["utils/watchdog.py", "managers/scheduler.py"])
    rec("watchdog_poll", wd, r"time\.sleep\((self\.watchdog_timeout\s*/\s*2)\)", "fires between 1.0x and 1.5x timeout")
    rec("watchdog_fire_msg", wd, r'(f?"[^"\n]*watchdog timeout[^"\n]*")')
    rec("watchdog_post_fire_sleep", wd, r"time\.sleep\((\d+)\)\s*\n\s*self\.parent_process\.send_signal\(signal\.SIGQUIT\)", "sleep before SIGQUIT to parent")
    rec("subprocess_watchdog_interval", wd, r"class SubprocessWatchdog.*?interval:\s*float\s*=\s*([\d.]+)", "rank-death poll period")
    rec("subprocess_watchdog_msg", wd, r'(f?"Subprocess \{name\}[^"\n]*crashed[^"\n]*")', "logged by the HTTP-server process when a rank exits non-zero")

    http = _find(srt, ["entrypoints/http_server.py"])
    rec("health_check_timeout", http, r'HEALTH_CHECK_TIMEOUT\s*=\s*int\(os\.getenv\("SGLANG_HEALTH_CHECK_TIMEOUT",\s*(\d+)\)\)', "/health and /health_generate 503 after this many s without any engine output")
    rec("health_handler", http, r'(@app\.get\("/health"\))')
    rec("health_fail_msg", http, r'(f?"Health check failed[^"\n]*")')

    ps = _find(srt, ["distributed/parallel_state.py"])
    rec("gloo_cpu_group_timeout", ps, r"gloo_timeout:\s*timedelta\s*=\s*timedelta\(seconds=([^)]+)\)", "CPU group used for the per-step TP request broadcast")
    rec("device_group_timeout_passthrough", ps, r"(_MODEL_PARALLEL_GROUP_TIMEOUT\s*=\s*timeout)", "None -> torch default NCCL pg timeout")

    sched = _find(srt, ["managers/scheduler.py"])
    rec("scheduler_exception_msg", sched, r'(f?"Scheduler hit an exception[^"\n]*")')
    rec("run_batch_def", sched, r"(def run_batch\()")
    rec("process_batch_result_def", sched, r"(def process_batch_result\()")
    rec("get_next_batch_to_run_def", sched, r"(def get_next_batch_to_run\()")
    rec("forward_ct_increment", sched, r"(self\.forward_ct\s*\+=\s*1)")

    tm = _find(srt, ["managers/tokenizer_manager.py"])
    rec("sigquit_handler_msg", tm, r'(f?"SIGQUIT received[^"\n]*")')

    out["findings"] = findings
    return out


def gpus():
    try:
        q = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
            text=True,
        )
        apps = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"], text=True
        )
        uuids = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True)
    except Exception as e:
        print(f"nvidia-smi failed: {e!r}")
        return 1
    uuid2idx = {}
    for line in uuids.strip().splitlines():
        idx, u = [x.strip() for x in line.split(",")]
        uuid2idx[u] = idx
    busy = {}
    for line in apps.strip().splitlines():
        if not line.strip():
            continue
        u, pid, mem = [x.strip() for x in line.split(",")]
        busy.setdefault(uuid2idx.get(u, u), []).append((pid, mem))
    print("idx | name | mem_used/total MiB | util% | compute procs (pid:MiB)")
    free = []
    for line in q.strip().splitlines():
        idx, name, used, total, util = [x.strip() for x in line.split(",")]
        procs = busy.get(idx, [])
        tag = "FREE" if not procs and int(used) < 1024 else "busy"
        if tag == "FREE":
            free.append(idx)
        print(f"{idx} | {name} | {used}/{total} | {util} | {procs} -> {tag}")
    print("free GPUs:", ",".join(free) if free else "none")
    return 0


def tree():
    import psutil

    pids = load_pids()
    if pids is None:
        print(f"no pids.json in {run_dir()}")
        return 1
    root = psutil.Process(pids["http_server"])
    print(f"{root.pid} {root.name()} [HTTP server + TokenizerManager]  status={root.status()}")
    rank_by_pid = {int(v): k for k, v in pids.get("ranks", {}).items()}
    for c in root.children(recursive=True):
        try:
            label = f"TP rank {rank_by_pid[c.pid]}" if c.pid in rank_by_pid else ("detokenizer" if c.pid == pids.get("detokenizer") else "")
            print(f"  {c.pid} ppid={c.ppid()} {c.name()} {label} status={c.status()} cmd={' '.join(c.cmdline())[:80]}")
        except psutil.NoSuchProcess:
            pass
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", action="store_true")
    ap.add_argument("--tree", action="store_true")
    ap.add_argument("--json", default=None, help="output path (default run/recon.json)")
    a = ap.parse_args()
    if a.gpus:
        return gpus()
    if a.tree:
        return tree()
    out = recon()
    path = Path(a.json) if a.json else run_dir() / "recon.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(f"sglang {out.get('sglang_version')} @ {out.get('sglang_path')} commit={out.get('sglang_git_commit')}")
    print(f"torch {out.get('torch_version')} nccl {out.get('nccl_version')} "
          f"default_pg_timeout={out.get('torch_default_pg_timeout_s')}s nccl_pg_timeout={out.get('torch_default_pg_nccl_timeout_s')}s")
    print("| finding | value | file:line |\n|---|---|---|")
    for k, v in out.get("findings", {}).items():
        loc = f"{v['file']}:{v['line']}" if v.get("line") else "NOT FOUND"
        print(f"| {k} | `{v['value']}` | {loc} |")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
