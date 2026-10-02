"""Wrapper launcher for SGLang Diffusion: stock server plus per-rank logs, PID map, step timestamps.

Same idea as ../launch_wrapped.py, adapted to sglang.multimodal_gen (verified
against 0.5.19): launch_server() spawns one process per GPU with
`target=run_scheduler_process` looked up from the launch_server module's globals,
so replacing that global with a wrapper is enough. Inside each rank the wrapper:
  1. redirects fd 1/2 to run/logs/rank<rank>.log
  2. writes run/pids/rank<rank>.pid
  3. wraps DenoisingStage._run_denoising_step -> run/logs/steps_rank<rank>.jsonl
     (one record per denoising step with the request's shape, so step durations
     can be binned by resolution/frames) and GPUWorker.execute_forward for
     request start/end on each rank
  4. calls the stock run_scheduler_process unchanged.
The HTTP server runs in the main process (uvicorn), which has no launch callback,
so a background thread writes run/pids.json once every rank PID file exists.

Usage: same CLI as `python -m sglang.multimodal_gen.runtime.launch_server`, e.g.
    python launch_wrapped_diff.py --model-path black-forest-labs/FLUX.1-dev --num-gpus 2 --sp-degree 2 --port 30000
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from common import JsonlWriter, ensure_run_dirs, logs_dir, now_ns, run_dir, wall  # noqa: E402
from launch_wrapped import _redirect_std_to  # noqa: E402


def _shape_of(batch):
    """Best-effort (w, h, frames, steps, rid) from a diffusion Req; field names vary by version."""
    sp = getattr(batch, "sampling_params", None)

    def g(*names):
        for obj in (batch, sp):
            for n in names:
                v = getattr(obj, n, None) if obj is not None else None
                if v is not None and not callable(v):
                    return v
        return None

    return {
        "w": g("width"), "h": g("height"), "nf": g("num_frames"),
        "n": g("num_inference_steps"), "rid": g("request_id", "rid", "id"),
    }


def _install_step_logger(rank: int) -> None:
    steps = JsonlWriter(logs_dir() / f"steps_rank{rank}.jsonl")
    steps.write({"ev": "patched", "pid": os.getpid(), "rank": rank})

    try:
        from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import DenoisingStage

        orig_step = DenoisingStage._run_denoising_step

        def _run_denoising_step(self, ctx, step, batch, server_args, *a, **k):
            t0 = now_ns()
            r = orig_step(self, ctx, step, batch, server_args, *a, **k)
            try:
                rec = {"ev": "ds", "i": getattr(step, "step_index", None), "t0_ns": t0, "t_ns": now_ns()}
                rec.update(_shape_of(batch))
                if rec.get("n") is None:
                    rec["n"] = getattr(ctx, "num_inference_steps", None)
                steps.write(rec)
            except Exception:
                pass
            return r

        DenoisingStage._run_denoising_step = _run_denoising_step
        steps.write({"ev": "hook", "target": "DenoisingStage._run_denoising_step", "ok": True})
    except Exception as e:
        steps.write({"ev": "hook", "target": "DenoisingStage._run_denoising_step", "ok": False, "error": repr(e)})

    try:
        from sglang.multimodal_gen.runtime.managers.gpu_worker import GPUWorker

        orig_fwd = GPUWorker.execute_forward

        def execute_forward(self, *a, **k):
            t0 = now_ns()
            steps.write({"ev": "req_start", "t_ns": t0})
            try:
                return orig_fwd(self, *a, **k)
            finally:
                steps.write({"ev": "req_end", "t0_ns": t0})

        GPUWorker.execute_forward = execute_forward
        steps.write({"ev": "hook", "target": "GPUWorker.execute_forward", "ok": True})
    except Exception as e:
        steps.write({"ev": "hook", "target": "GPUWorker.execute_forward", "ok": False, "error": repr(e)})


def run_scheduler_process_wrapped(local_rank, rank, master_port, server_args, *args, **kwargs):
    """Drop-in for sglang.multimodal_gen.runtime.managers.gpu_worker.run_scheduler_process."""
    ensure_run_dirs()
    _redirect_std_to(str(logs_dir() / f"rank{rank}.log"))
    with open(run_dir() / "pids" / f"rank{rank}.pid", "w") as f:
        f.write(str(os.getpid()))
    print(f"[ft-detect] rank{rank} local_rank={local_rank} pid={os.getpid()} t_ns={now_ns()} wall={wall():.3f}", flush=True)
    try:
        _install_step_logger(rank)
    except Exception as e:
        print(f"[ft-detect] step logger install failed: {e!r}", flush=True)

    from sglang.multimodal_gen.runtime.managers.gpu_worker import run_scheduler_process

    return run_scheduler_process(local_rank, rank, master_port, server_args, *args, **kwargs)


def _pids_writer(num_gpus: int) -> None:
    import psutil

    deadline = time.time() + 7200
    while time.time() < deadline:
        ranks = {}
        for i in range(num_gpus):
            p = run_dir() / "pids" / f"rank{i}.pid"
            if p.exists():
                try:
                    ranks[str(i)] = int(p.read_text().strip())
                except ValueError:
                    pass
        if len(ranks) == num_gpus:
            me = psutil.Process(os.getpid())
            others = {}
            for c in me.children(recursive=False):
                if c.pid in ranks.values():
                    continue
                try:
                    others[str(c.pid)] = c.name() + " " + " ".join(c.cmdline()[-2:])
                except Exception:
                    others[str(c.pid)] = "?"
            rec = {"http_server": os.getpid(), "tp_size": num_gpus, "stack": "sglang-diffusion", "ranks": ranks,
                   "detokenizer": None, "other_children": others, "t_ready_ns": now_ns(), "wall_ready": wall(),
                   "argv": sys.argv[1:]}
            with open(run_dir() / "pids.json", "w") as f:
                json.dump(rec, f, indent=2)
            print(f"[ft-detect] wrote {run_dir() / 'pids.json'}: {rec}", flush=True)
            return
        time.sleep(1)


def main() -> None:
    import sglang.multimodal_gen.runtime.launch_server as ls
    from sglang.multimodal_gen.runtime.server_args.server_args import prepare_server_args

    ensure_run_dirs()
    server_args = prepare_server_args(sys.argv[1:])
    num_gpus = int(getattr(server_args, "num_gpus", 1))
    with open(run_dir() / "launch.json", "w") as f:
        json.dump({"t_launch_ns": now_ns(), "wall_launch": wall(), "pid": os.getpid(), "argv": sys.argv[1:],
                   "stack": "sglang-diffusion"}, f)

    # launch_server() reads this name from its module globals when it spawns workers.
    assert hasattr(ls, "run_scheduler_process"), "launch_server module has no run_scheduler_process global; version drift"
    ls.run_scheduler_process = run_scheduler_process_wrapped
    threading.Thread(target=_pids_writer, args=(num_gpus,), daemon=True).start()

    try:
        if hasattr(ls, "dispatch_launch"):
            ls.dispatch_launch(server_args)
        else:
            ls.launch_server(server_args)
    finally:
        try:
            from sglang.multimodal_gen.runtime.utils.process import kill_process_tree

            kill_process_tree(os.getpid(), include_parent=False)
        except Exception:
            pass


if __name__ == "__main__":
    main()
