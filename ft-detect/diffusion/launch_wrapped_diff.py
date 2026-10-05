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


# ---- Phase 3c: containment probe (measurement only) -----------------------------------------
# FT_ABORT_PROBE=1 starts a thread per rank that watches our own progress records (stage and
# step hooks update _PROGRESS). If a request is in flight and no quantum has completed for
# FT_ABORT_DEADLINE_S seconds, the thread aborts the torch process groups
# (torch.distributed.distributed_c10d._abort_process_group -> ncclCommAbort) and timestamps
# every phase so the cost of containment can be measured: deadline miss -> abort call ->
# abort return -> exception surfacing in the main thread -> request end -> loop alive again.
# This is a probe for the experiment, not a detector for production: the deadline is a
# constant, and nothing marks the replica unroutable.
import threading as _threading

_PROGRESS = {"last_ns": 0, "in_request": False, "req_start_ns": 0, "fired": False, "lock": _threading.Lock()}


def _progress_touch():
    _PROGRESS["last_ns"] = now_ns()


def _abort_all_groups(log) -> dict:
    """Abort every torch process group this rank holds; returns per-attempt timings."""
    import torch
    import torch.distributed as dist
    from torch.distributed import distributed_c10d as c10d

    out = {"attempts": []}
    if not dist.is_initialized():
        out["error"] = "torch.distributed not initialized"
        return out
    mode = os.environ.get("FT_ABORT_MODE", "all")   # all | sp | world | exit
    if mode == "exit":
        # process-level containment: give up on the collective and leave; measures how fast the
        # GPU is released and how the peer / HTTP side react to a vanished rank
        out["attempts"].append({"group": "exit", "how": "os._exit(3)", "ok": True, "t0_ns": now_ns(), "t_ns": now_ns()})
        log("[ft-detect] FT_ABORT_PROBE abort returned after 0.0 ms: exiting rank process (FT_ABORT_MODE=exit)")
        sys.stdout.flush()
        os._exit(3)
    groups = []
    if mode in ("sp", "world"):
        try:
            from sglang.multimodal_gen.runtime.distributed import parallel_state as ps

            g = (ps.get_sp_group() if mode == "sp" else ps.get_world_group()).device_group
            groups.append((mode, g))
        except Exception as e:
            out["error"] = f"group lookup failed: {e!r}"
    for name, g in groups or [("all", None)]:
        t0 = now_ns()
        try:
            if hasattr(c10d, "_abort_process_group"):
                c10d._abort_process_group(g)
                how = "c10d._abort_process_group"
            else:
                (g or dist.group.WORLD).abort()
                how = "ProcessGroup.abort"
            out["attempts"].append({"group": name, "how": how, "ok": True, "t0_ns": t0, "t_ns": now_ns()})
        except Exception as e:
            out["attempts"].append({"group": name, "ok": False, "error": repr(e), "t0_ns": t0, "t_ns": now_ns()})
    return out


def _abort_probe_thread(steps: "JsonlWriter", rank: int) -> None:
    import torch

    deadline_s = float(os.environ.get("FT_ABORT_DEADLINE_S", "10"))
    while True:
        time.sleep(0.05)
        if not _PROGRESS["in_request"] or _PROGRESS["fired"]:
            continue
        last = max(_PROGRESS["last_ns"], _PROGRESS["req_start_ns"])
        if last and (now_ns() - last) / 1e9 >= deadline_s:
            with _PROGRESS["lock"]:
                if _PROGRESS["fired"]:
                    continue
                _PROGRESS["fired"] = True
            t_miss = now_ns()
            mem0 = torch.cuda.memory_allocated() if torch.cuda.is_available() else None
            print(f"[ft-detect] FT_ABORT_PROBE deadline miss: no progress for {deadline_s}s (rank{rank}); aborting process groups", flush=True)
            steps.write({"ev": "abort_probe", "phase": "deadline_miss", "deadline_s": deadline_s, "t_ns": t_miss,
                         "last_progress_ns": last, "cuda_mem_alloc": mem0})
            if os.environ.get("FT_ABORT_MODE", "all") == "exit":
                steps.write({"ev": "abort_probe", "phase": "abort_returned", "t_ns": now_ns(),
                             "result": {"attempts": [{"group": "exit", "how": "os._exit(3)", "ok": True}]}, "cuda_mem_alloc": mem0})
            res = _abort_all_groups(print)
            t_ret = now_ns()
            print(f"[ft-detect] FT_ABORT_PROBE abort returned after {(t_ret - t_miss) / 1e6:.1f} ms: {res}", flush=True)
            steps.write({"ev": "abort_probe", "phase": "abort_returned", "t_ns": t_ret, "result": res,
                         "cuda_mem_alloc": torch.cuda.memory_allocated() if torch.cuda.is_available() else None})
            # keep watching: record whether the process is still alive and the loop resumes
            for i in range(600):
                time.sleep(0.5)
                if not _PROGRESS["in_request"]:
                    steps.write({"ev": "abort_probe", "phase": "loop_idle_again", "t_ns": now_ns(),
                                 "cuda_mem_alloc": torch.cuda.memory_allocated() if torch.cuda.is_available() else None})
                    break
            return


# ---- Phase 3b-1: trajectory save / resume across processes and SP degrees --------------------
# Save:   FT_TRAJ_SAVE_DIR=<dir> [FT_TRAJ_SAVE_STEP=k]  -> <dir>/traj_step<k>_rank<r>.pt after step k,
#         and always <dir>/final_rank<r>.pt after the last step (full, gathered latents).
# Resume: FT_TRAJ_RESUME=<traj file> FT_TRAJ_RESUME_STEP=k FT_TRAJ_MODE=full|lower
#         steps 0..k are skipped (no compute); before step k+1 the saved latents and scheduler
#         state are injected ('full' restores the multi-step solver history, 'lower' resets it so
#         the solver restarts at low order). Timings go to steps_rank<r>.jsonl as ev 'traj'.
_TRAJ = {"resumed": False, "t_resume_ns": None, "first_step_done": False}
_SCHED_ATTRS = ("_step_index", "_begin_index", "model_outputs", "last_sample", "timestep_list",
                "lower_order_nums", "this_order")


def _gather_full(t, batch, server_args):
    """Full (unsharded) copy of a per-rank latent-shaped tensor; a collective when sharded."""
    import torch

    if not isinstance(t, torch.Tensor):
        return t
    if getattr(batch, "did_sp_shard_latents", False):
        try:
            return server_args.pipeline_config.gather_latents_for_sp(t, batch)
        except TypeError:
            return server_args.pipeline_config.gather_latents_for_sp(t)
    return t


def _traj_bundle(ctx, step, batch, server_args):
    import torch

    sch = ctx.scheduler
    b = {"step_index": int(step.step_index), "num_inference_steps": int(ctx.num_inference_steps),
         "latents": _gather_full(ctx.latents, batch, server_args).detach().clone(),
         "latents_local_shape": list(ctx.latents.shape),
         "did_sp_shard_latents": bool(getattr(batch, "did_sp_shard_latents", False)),
         "scheduler_class": type(sch).__name__, "scheduler": {}}
    for name in _SCHED_ATTRS:
        if not hasattr(sch, name):
            continue
        v = getattr(sch, name)
        if isinstance(v, torch.Tensor):
            v = _gather_full(v, batch, server_args).detach().clone()
        elif isinstance(v, list):
            v = [(_gather_full(x, batch, server_args).detach().clone() if isinstance(x, torch.Tensor) else x) for x in v]
        b["scheduler"][name] = v
    b["timesteps"] = ctx.timesteps.detach().clone()
    b["rid"] = _shape_of(batch).get("rid")
    return b


def _traj_save(ctx, step, batch, server_args, rank, tag):
    import torch

    d = os.environ.get("FT_TRAJ_SAVE_DIR")
    if not d:
        return None
    os.makedirs(d, exist_ok=True)
    t0 = now_ns()
    b = _traj_bundle(ctx, step, batch, server_args)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = now_ns()
    path = os.path.join(d, f"{tag}_rank{rank}.pt")
    torch.save({k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in b.items()} | {
        "scheduler": {k: (v.cpu() if isinstance(v, torch.Tensor) else [x.cpu() if isinstance(x, torch.Tensor) else x for x in v] if isinstance(v, list) else v)
                      for k, v in b["scheduler"].items()}}, path)
    t2 = now_ns()
    return {"path": path, "t_bundle_ms": (t1 - t0) / 1e6, "t_save_ms": (t2 - t1) / 1e6, "bytes": os.path.getsize(path)}


def _traj_restore(ctx, batch, device):
    """Inject the saved boundary state into ctx/scheduler before the first resumed step."""
    import torch

    path = os.environ["FT_TRAJ_RESUME"]
    mode = os.environ.get("FT_TRAJ_MODE", "full")
    t0 = now_ns()
    saved = torch.load(path, map_location="cpu", weights_only=False)
    t1 = now_ns()
    lat = saved["latents"].to(device=device, dtype=ctx.latents.dtype)
    notes = []
    if list(lat.shape) != list(ctx.latents.shape):
        # generic image configs pad the sequence before sharding; crop back to the live shape
        sl = tuple(slice(0, n) for n in ctx.latents.shape)
        notes.append(f"shape {list(lat.shape)} -> {list(ctx.latents.shape)}")
        lat = lat[sl]
    ctx.latents = lat.contiguous()
    sch = ctx.scheduler
    for name, v in saved["scheduler"].items():
        # Solver state is injected even when the fresh scheduler has not created the attribute
        # yet (UniPC sets this_order inside its first step(); a restored lower_order_nums > 0
        # makes it read this_order on the very next step).
        if mode == "lower" and name in ("model_outputs", "last_sample", "timestep_list", "lower_order_nums", "this_order"):
            cur = getattr(sch, name, None)
            if name == "model_outputs" and isinstance(cur, list):
                setattr(sch, name, [None] * len(cur))
            elif name == "timestep_list" and isinstance(cur, list):
                setattr(sch, name, [None] * len(cur))
            elif name == "lower_order_nums":
                setattr(sch, name, 0)
            elif name == "last_sample":
                setattr(sch, name, None)
            continue
        if isinstance(v, torch.Tensor):
            v = v.to(device=device)
        elif isinstance(v, list):
            v = [x.to(device=device) if isinstance(x, torch.Tensor) else x for x in v]
        setattr(sch, name, v)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t2 = now_ns()
    return {"path": path, "mode": mode, "saved_step": saved["step_index"], "t_load_ms": (t1 - t0) / 1e6,
            "t_inject_ms": (t2 - t1) / 1e6, "notes": notes, "scheduler_class": saved.get("scheduler_class")}


# ---- Phase 3a: resume-state inventory at a denoising-step boundary -------------------------
# Enabled with FT_INVENTORY=1; taken once per request on every rank right after step
# FT_INVENTORY_STEP (default 2) completes, i.e. when ctx.latents is x_{t-1} at a boundary.

def _tensor_info(t, with_checksum: bool):
    import torch

    if not isinstance(t, torch.Tensor):
        return None
    info = {"shape": list(t.shape), "dtype": str(t.dtype).replace("torch.", ""), "device": str(t.device),
            "nbytes": t.numel() * t.element_size()}
    if with_checksum and t.numel() > 0 and t.is_floating_point():
        # one device sync per tensor; only used once per request
        info["checksum"] = float(t.detach().double().sum().item())
    return info


def _walk_tensors(obj, prefix: str, out: dict, with_checksum: bool, depth: int = 0):
    """Collect tensor infos from tensors, lists/tuples and dicts (bounded depth)."""
    import torch

    if depth > 3 or obj is None:
        return
    if isinstance(obj, torch.Tensor):
        out[prefix] = _tensor_info(obj, with_checksum)
    elif isinstance(obj, (list, tuple)):
        for i, x in enumerate(obj[:16]):
            _walk_tensors(x, f"{prefix}[{i}]", out, with_checksum, depth + 1)
    elif isinstance(obj, dict):
        for k, v in list(obj.items())[:32]:
            _walk_tensors(v, f"{prefix}.{k}", out, with_checksum, depth + 1)


def _generator_info(gen):
    import torch

    gens = gen if isinstance(gen, (list, tuple)) else [gen]
    out = []
    for g in gens:
        if isinstance(g, torch.Generator):
            try:
                st = g.get_state()
                out.append({"device": str(g.device), "state_bytes": int(st.numel() * st.element_size())})
            except Exception as e:
                out.append({"device": str(g.device), "error": repr(e)})
    return out


def _resume_bundle(ctx, step, batch):
    """Minimum state to continue a request from this boundary; what a checkpoint would carry."""
    import torch

    b = {"latents": ctx.latents, "step_index": step.step_index}
    sch = ctx.scheduler
    for name in ("model_outputs", "last_sample", "timestep_list"):   # multi-step solver history
        v = getattr(sch, name, None)
        if isinstance(v, torch.Tensor) or (isinstance(v, list) and any(isinstance(x, torch.Tensor) for x in v)):
            b[f"scheduler.{name}"] = v
    for name in ("z", "reserved_frames_mask", "guidance"):
        v = getattr(ctx, name, None)
        if isinstance(v, torch.Tensor):
            b[f"ctx.{name}"] = v
    for name in ("prompt_embeds", "negative_prompt_embeds", "pooled_embeds", "neg_pooled_embeds",
                 "prompt_attention_mask", "negative_attention_mask", "prompt_embeds_mask",
                 "negative_prompt_embeds_mask", "image_embeds", "image_latent", "clip_embedding_pos",
                 "clip_embedding_neg"):
        v = getattr(batch, name, None)
        if v is not None and v != []:
            b[f"req.{name}"] = v
    gen = getattr(batch, "generator", None)
    if gen is not None:
        gens = gen if isinstance(gen, (list, tuple)) else [gen]
        b["generator_state"] = [g.get_state() for g in gens if isinstance(g, torch.Generator)]
    return b


def _measure_checkpoint(bundle):
    """T_pin: device -> pinned host copy; T_save: torch.save of the host copy; sizes in bytes."""
    import io
    import time

    import torch

    def to_host(x):
        if isinstance(x, torch.Tensor):
            if x.is_cuda:
                h = torch.empty_like(x, device="cpu", pin_memory=True)
                h.copy_(x, non_blocking=False)
                return h
            return x.clone()
        if isinstance(x, (list, tuple)):
            return [to_host(e) for e in x]
        if isinstance(x, dict):
            return {k: to_host(v) for k, v in x.items()}
        return x

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    host = to_host(bundle)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    buf = io.BytesIO()
    torch.save(host, buf)
    t2 = time.perf_counter()
    return {"t_pin_ms": (t1 - t0) * 1e3, "t_save_ms": (t2 - t1) * 1e3, "serialized_bytes": buf.getbuffer().nbytes}


def _inventory_record(ctx, step, batch, rank: int) -> dict:
    tensors: dict = {}
    _walk_tensors(ctx.latents, "latents", tensors, True)
    for name in ("z", "reserved_frames_mask", "guidance", "timesteps"):
        _walk_tensors(getattr(ctx, name, None), f"ctx.{name}", tensors, False)
    for name in ("image_kwargs", "pos_cond_kwargs", "neg_cond_kwargs", "extra_step_kwargs"):
        _walk_tensors(getattr(ctx, name, None), f"ctx.{name}", tensors, False)
    for name in ("prompt_embeds", "negative_prompt_embeds", "pooled_embeds", "neg_pooled_embeds",
                 "prompt_attention_mask", "image_embeds", "image_latent", "clip_embedding_pos", "latents"):
        _walk_tensors(getattr(batch, name, None), f"req.{name}", tensors, name == "latents")
    sch = ctx.scheduler
    for name in ("model_outputs", "last_sample", "sigmas", "timesteps"):
        _walk_tensors(getattr(sch, name, None), f"scheduler.{name}", tensors, False)
    bundle = _resume_bundle(ctx, step, batch)
    import torch

    def nbytes(x):
        if isinstance(x, torch.Tensor):
            return x.numel() * x.element_size()
        if isinstance(x, (list, tuple)):
            return sum(nbytes(e) for e in x)
        if isinstance(x, dict):
            return sum(nbytes(v) for v in x.values())
        return 0

    rec = {
        "ev": "inventory", "rank": rank, "step_index": getattr(step, "step_index", None),
        "num_inference_steps": getattr(ctx, "num_inference_steps", None),
        "did_sp_shard_latents": bool(getattr(batch, "did_sp_shard_latents", False)),
        "scheduler_class": type(sch).__name__,
        "scheduler_step_index": getattr(sch, "_step_index", None),
        "generator": _generator_info(getattr(batch, "generator", None)),
        "tensors": tensors,
        "bundle_bytes": {k: nbytes(v) for k, v in bundle.items()},
        "bundle_total_bytes": nbytes(bundle),
    }
    rec.update(_shape_of(batch))
    try:
        rec["checkpoint"] = _measure_checkpoint(bundle)
    except Exception as e:
        rec["checkpoint_error"] = repr(e)
    return rec


def _install_step_logger(rank: int) -> None:
    steps = JsonlWriter(logs_dir() / f"steps_rank{rank}.jsonl")
    steps.write({"ev": "patched", "pid": os.getpid(), "rank": rank})

    try:
        from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import DenoisingStage

        orig_step = DenoisingStage._run_denoising_step

        inventory_on = os.environ.get("FT_INVENTORY", "0") == "1"
        inventory_step = int(os.environ.get("FT_INVENTORY_STEP", "2"))
        traj_save_step = int(os.environ["FT_TRAJ_SAVE_STEP"]) if os.environ.get("FT_TRAJ_SAVE_STEP") else None
        traj_resume = os.environ.get("FT_TRAJ_RESUME")
        traj_resume_step = int(os.environ.get("FT_TRAJ_RESUME_STEP", "-1"))

        def _run_denoising_step(self, ctx, step, batch, server_args, *a, **k):
            si = getattr(step, "step_index", None)
            if traj_resume and si is not None and si <= traj_resume_step:
                steps.write({"ev": "ds_skip", "i": si})
                return None
            if traj_resume and si == traj_resume_step + 1 and not _TRAJ["resumed"]:
                try:
                    info = _traj_restore(ctx, batch, ctx.latents.device)
                    _TRAJ.update({"resumed": True, "t_resume_ns": now_ns()})
                    steps.write({"ev": "traj", "phase": "restored", **info})
                except Exception as e:
                    steps.write({"ev": "traj", "phase": "restore_failed", "error": repr(e)[:300]})
                    raise
            t0 = now_ns()
            try:
                r = orig_step(self, ctx, step, batch, server_args, *a, **k)
            except BaseException as e:
                steps.write({"ev": "step_exception", "i": getattr(step, "step_index", None), "t0_ns": t0,
                             "t_ns": now_ns(), "error": repr(e)[:300]})
                raise
            _progress_touch()
            try:
                rec = {"ev": "ds", "i": getattr(step, "step_index", None), "t0_ns": t0, "t_ns": now_ns()}
                rec.update(_shape_of(batch))
                if rec.get("n") is None:
                    rec["n"] = getattr(ctx, "num_inference_steps", None)
                steps.write(rec)
            except Exception:
                pass
            if inventory_on and getattr(step, "step_index", None) == inventory_step:
                try:
                    steps.write(_inventory_record(ctx, step, batch, rank))
                except Exception as e:
                    steps.write({"ev": "inventory_error", "error": repr(e)})
            if traj_resume and _TRAJ["resumed"] and not _TRAJ["first_step_done"]:
                _TRAJ["first_step_done"] = True
                steps.write({"ev": "traj", "phase": "first_resumed_step", "i": si, "t_step_ms": (now_ns() - t0) / 1e6})
            n_steps = getattr(ctx, "num_inference_steps", None)
            if os.environ.get("FT_TRAJ_SAVE_DIR"):
                try:
                    if traj_save_step is not None and si == traj_save_step:
                        steps.write({"ev": "traj", "phase": "saved", "i": si, **_traj_save(ctx, step, batch, server_args, rank, f"traj_step{si}")})
                    if n_steps is not None and si == n_steps - 1:
                        steps.write({"ev": "traj", "phase": "final_saved", "i": si, **_traj_save(ctx, step, batch, server_args, rank, "final")})
                        if _TRAJ["resumed"]:
                            steps.write({"ev": "traj", "phase": "remaining_done", "t_remaining_ms": (now_ns() - _TRAJ["t_resume_ns"]) / 1e6})
                except Exception as e:
                    steps.write({"ev": "traj", "phase": "save_failed", "error": repr(e)[:300]})
            return r

        DenoisingStage._run_denoising_step = _run_denoising_step
        steps.write({"ev": "hook", "target": "DenoisingStage._run_denoising_step", "ok": True})
    except Exception as e:
        steps.write({"ev": "hook", "target": "DenoisingStage._run_denoising_step", "ok": False, "error": repr(e)})

    try:
        from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage

        orig_call = PipelineStage.__call__

        def __call__(self, batch, server_args, *a, **k):
            t0 = now_ns()
            err = None
            try:
                return orig_call(self, batch, server_args, *a, **k)
            except BaseException as e:
                err = repr(e)[:300]
                raise
            finally:
                _progress_touch()
                try:
                    rec = {"ev": "stage", "name": getattr(self, "_registered_stage_name", None) or type(self).__name__,
                           "cls": type(self).__name__, "t0_ns": t0, "t_ns": now_ns()}
                    if err:
                        rec["error"] = err
                    rec.update(_shape_of(batch))
                    steps.write(rec)
                except Exception:
                    pass

        PipelineStage.__call__ = __call__
        steps.write({"ev": "hook", "target": "PipelineStage.__call__", "ok": True})
    except Exception as e:
        steps.write({"ev": "hook", "target": "PipelineStage.__call__", "ok": False, "error": repr(e)})

    try:
        from sglang.multimodal_gen.runtime.managers.gpu_worker import GPUWorker

        orig_fwd = GPUWorker.execute_forward

        def execute_forward(self, *a, **k):
            t0 = now_ns()
            steps.write({"ev": "req_start", "t_ns": t0})
            _PROGRESS.update({"in_request": True, "req_start_ns": t0, "last_ns": t0})
            err = None
            try:
                return orig_fwd(self, *a, **k)
            except BaseException as e:
                err = repr(e)[:300]
                raise
            finally:
                _PROGRESS["in_request"] = False
                steps.write({"ev": "req_end", "t0_ns": t0, "error": err})

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
    if os.environ.get("FT_ABORT_PROBE", "0") == "1":
        _threading.Thread(target=_abort_probe_thread, args=(JsonlWriter(logs_dir() / f"steps_rank{rank}.jsonl"), rank),
                          daemon=True, name="ft-abort-probe").start()
        print(f"[ft-detect] FT_ABORT_PROBE armed on rank{rank}: deadline {os.environ.get('FT_ABORT_DEADLINE_S', '10')}s, "
              f"mode {os.environ.get('FT_ABORT_MODE', 'all')}", flush=True)

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
