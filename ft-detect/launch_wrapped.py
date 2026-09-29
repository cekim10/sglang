"""Wrapper launcher: stock SGLang server plus per-rank logs, PID map, step timestamps.

Nothing in site-packages is modified. This script calls launch_server() with a
replacement scheduler entrypoint that, inside each spawned rank process:
  1. redirects that process's fd 1/2 to  run/logs/rank<tp_rank>.log
  2. writes its PID to                   run/pids/rank<tp_rank>.pid
  3. wraps Scheduler.run_batch / process_batch_result to append one record per
     step to                             run/logs/steps_rank<tp_rank>.jsonl
  4. calls the stock run_scheduler_process unchanged.

When the HTTP server reports ready, launch_callback writes run/pids.json.

Usage: identical CLI to `python -m sglang.launch_server`, e.g.
    python launch_wrapped.py --model-path <m> --tp-size 4 --port 30000 --watchdog-timeout 300
"""

from __future__ import annotations

import inspect
import json
import os
import sys
import time

# Make common.py importable in the spawned children too (they re-import this
# module as __mp_main__ from its file path, so sys.path is the same).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import JsonlWriter, ensure_run_dirs, logs_dir, now_ns, run_dir, wall  # noqa: E402


def _redirect_std_to(path: str) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    os.close(fd)
    sys.stdout = open(1, "w", buffering=1, closefd=False)
    sys.stderr = open(2, "w", buffering=1, closefd=False)


def _install_step_logger(tp_rank: int) -> None:
    from sglang.srt.managers.scheduler import Scheduler

    steps = JsonlWriter(logs_dir() / f"steps_rank{tp_rank}.jsonl")
    steps.write({"ev": "patched", "pid": os.getpid(), "tp_rank": tp_rank})

    orig_run_batch = Scheduler.run_batch
    orig_pbr = Scheduler.process_batch_result

    def run_batch(self, batch, *a, **k):
        t0 = now_ns()
        result = orig_run_batch(self, batch, *a, **k)
        try:
            steps.write(
                {
                    "ev": "rb",
                    "fc": getattr(self, "forward_ct", -1),
                    "bs": batch.batch_size() if batch is not None else 0,
                    "mode": str(getattr(batch, "forward_mode", "")),
                    "t0_ns": t0,
                    "t_ns": now_ns(),
                }
            )
        except Exception:
            pass
        return result

    def process_batch_result(self, batch, result, *a, **k):
        r = orig_pbr(self, batch, result, *a, **k)
        try:
            steps.write({"ev": "pbr", "fc": getattr(self, "forward_ct", -1)})
        except Exception:
            pass
        return r

    Scheduler.run_batch = run_batch
    Scheduler.process_batch_result = process_batch_result


def run_scheduler_process_wrapped(server_args, port_args, gpu_id, tp_rank, *args, **kwargs):
    """Drop-in for sglang.srt.managers.scheduler.run_scheduler_process."""
    ensure_run_dirs()
    _redirect_std_to(str(logs_dir() / f"rank{tp_rank}.log"))
    with open(run_dir() / "pids" / f"rank{tp_rank}.pid", "w") as f:
        f.write(str(os.getpid()))
    print(
        f"[ft-detect] rank{tp_rank} pid={os.getpid()} gpu_id={gpu_id} "
        f"t_ns={now_ns()} wall={wall():.3f}",
        flush=True,
    )
    try:
        _install_step_logger(tp_rank)
    except Exception as e:  # never let instrumentation break the server
        print(f"[ft-detect] step logger install failed: {e!r}", flush=True)

    from sglang.srt.managers.scheduler import run_scheduler_process

    return run_scheduler_process(server_args, port_args, gpu_id, tp_rank, *args, **kwargs)


def _write_pids_json(tp_size: int) -> None:
    import psutil

    ranks = {}
    for i in range(tp_size):
        p = run_dir() / "pids" / f"rank{i}.pid"
        if p.exists():
            ranks[str(i)] = int(p.read_text().strip())
    me = psutil.Process(os.getpid())
    children = {c.pid: c for c in me.children(recursive=False)}
    others = {}
    for pid, c in children.items():
        if pid in ranks.values():
            continue
        try:
            others[str(pid)] = c.name() + " " + " ".join(c.cmdline()[-2:])
        except Exception:
            others[str(pid)] = "?"
    detok = None
    for pid, desc in others.items():
        if "resource_tracker" not in desc:
            detok = int(pid)
    rec = {
        "http_server": os.getpid(),
        "tp_size": tp_size,
        "ranks": ranks,
        "detokenizer": detok,
        "other_children": others,
        "t_ready_ns": now_ns(),
        "wall_ready": wall(),
        "argv": sys.argv[1:],
    }
    with open(run_dir() / "pids.json", "w") as f:
        json.dump(rec, f, indent=2)
    print(f"[ft-detect] wrote {run_dir() / 'pids.json'}: {rec}", flush=True)


def main() -> None:
    from sglang.srt.entrypoints.http_server import launch_server
    from sglang.srt.server_args import prepare_server_args

    try:
        from sglang.srt.plugins import load_plugins

        load_plugins()
    except Exception:
        pass

    ensure_run_dirs()
    server_args = prepare_server_args(sys.argv[1:])
    tp_size = int(getattr(server_args, "tp_size", 1))
    with open(run_dir() / "launch.json", "w") as f:
        json.dump(
            {"t_launch_ns": now_ns(), "wall_launch": wall(), "pid": os.getpid(), "argv": sys.argv[1:]},
            f,
        )

    kwargs = {"run_scheduler_process_func": run_scheduler_process_wrapped}
    if "launch_callback" in inspect.signature(launch_server).parameters:
        kwargs["launch_callback"] = lambda: _write_pids_json(tp_size)
    else:
        print("[ft-detect] launch_server has no launch_callback; write pids.json from launch.sh", flush=True)
    try:
        launch_server(server_args, **kwargs)
    finally:
        # same teardown as sglang/launch_server.py
        try:
            from sglang.srt.utils import kill_process_tree

            kill_process_tree(os.getpid(), include_parent=False)
        except Exception:
            pass


if __name__ == "__main__":
    main()
