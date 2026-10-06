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

Test 3 (FT_CONTAIN=1): failure-contained execution. Collectives are issued async and polled
under FT_CONTAIN_DEADLINE_S (default 5 s); a miss inside a request aborts the process groups,
shrinks every group coordinator to this rank, disables the DiT's internal sequence shard,
kills the peer and re-runs the same step at SP=1 in the same process (see README, Test 3).
FT_FAIL_STEP=k FT_FAIL_RANK=r FT_FAIL_REQ=n [FT_FAIL_SHAPE=WxHxF]: rank r SIGSTOPs itself at
step k of its n-th non-warmup request of that shape (the deterministic injection used by
run_contain.sh; the shape filter keeps probe/other requests from advancing the count).

Usage: same CLI as `python -m sglang.multimodal_gen.runtime.launch_server`, e.g.
    python launch_wrapped_diff.py --model-path black-forest-labs/FLUX.1-dev --num-gpus 2 --sp-degree 2 --port 30000
"""

from __future__ import annotations

import json
import os
import signal
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


_COEXIST = {"bufs": None, "stream": None}


def _main_thread_stack() -> str:
    import threading
    import traceback

    frames = sys._current_frames()
    main_id = threading.main_thread().ident
    f = frames.get(main_id)
    return "".join(traceback.format_stack(f)[-12:]) if f is not None else "<no main frame>"


def _side_stream_bench(n_iter: int = 30, size: int = 4096, log=None, reuse: bool = True) -> dict:
    """Independent GPU work on a side CUDA stream while the main thread may be stuck, in phases
    that separate the suspects: (A) in-place kernel on a pre-allocated buffer (no allocator, one
    launch), (B) matmul into a pre-allocated output (cuBLAS, no allocator), (C) fresh allocation.
    Every blocking call is announced through `log` first. Synchronisation is per-stream-event only."""
    import torch

    if not torch.cuda.is_available():
        return {"error": "no cuda"}
    say = log or (lambda m: None)
    dev = torch.device("cuda", torch.cuda.current_device())
    out = {}
    tw = time.perf_counter
    if reuse and _COEXIST["bufs"] is not None:
        stream = _COEXIST["stream"]; a, b, c, d, h = _COEXIST["bufs"]; out["buffers"] = "reused"
    else:
        say("coexist: creating stream"); stream = torch.cuda.Stream(device=dev)
        with torch.cuda.stream(stream):
            say("coexist: cudaMalloc a,b,c"); t = tw()
            a = torch.randn(size, size, device=dev, dtype=torch.bfloat16); b = torch.randn(size, size, device=dev, dtype=torch.bfloat16)
            c = torch.empty(size, size, device=dev, dtype=torch.bfloat16); out["alloc_abc_ms"] = (tw() - t) * 1e3
            say("coexist: cudaMalloc d (256 MiB)"); t = tw(); d = torch.empty(256 * 1024 * 1024 // 2, device=dev, dtype=torch.bfloat16); out["alloc_256MiB_ms"] = (tw() - t) * 1e3
            say("coexist: cudaHostAlloc h (64 MiB pinned)"); t = tw(); h = torch.empty(64 * 1024 * 1024 // 2, dtype=torch.bfloat16, pin_memory=True); out["pin_64MiB_ms"] = (tw() - t) * 1e3
            torch.matmul(a, b, out=c)   # warm cuBLAS workspace on this stream
        _COEXIST.update({"bufs": (a, b, c, d, h), "stream": stream}); out["buffers"] = "allocated"
    with torch.cuda.stream(stream):
        # (A) one in-place kernel, no allocation
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        say("coexist: A launch in-place zero_ (no alloc)"); t = tw(); e0.record(stream); d.zero_(); e1.record(stream); out["A_launch_ms"] = (tw() - t) * 1e3
        say("coexist: A event sync"); t = tw(); e1.synchronize(); out["A_sync_ms"] = (tw() - t) * 1e3; out["A_kernel_ms"] = e0.elapsed_time(e1)
        # (B) matmuls into a pre-allocated output, no allocation
        e2 = torch.cuda.Event(enable_timing=True); e3 = torch.cuda.Event(enable_timing=True)
        say("coexist: B launch matmul(out=) x%d (no alloc)" % n_iter); t = tw(); e2.record(stream)
        for _ in range(n_iter):
            torch.matmul(a, b, out=c)
        e3.record(stream); out["B_launch_ms"] = (tw() - t) * 1e3
        say("coexist: B event sync"); t = tw(); e3.synchronize(); out["B_sync_ms"] = (tw() - t) * 1e3; out["matmul_ms"] = e2.elapsed_time(e3) / n_iter
        # (C) allocation through the caching allocator
        say("coexist: C allocate 64 MiB (caching allocator)"); t = tw(); x = torch.empty(64 * 1024 * 1024 // 2, device=dev, dtype=torch.bfloat16); out["C_alloc_ms"] = (tw() - t) * 1e3
        e4 = torch.cuda.Event(enable_timing=True); e5 = torch.cuda.Event(enable_timing=True)
        say("coexist: C d2h 64 MiB"); e4.record(stream); h.copy_(d[: h.numel()], non_blocking=True); e5.record(stream); e5.synchronize(); out["d2h_64MiB_ms"] = e4.elapsed_time(e5)
        del x
        say("coexist: done")
    out["mem_alloc_MiB"] = torch.cuda.memory_allocated() / 2**20
    return out


def _abort_all_groups(log) -> dict:
    """Abort every torch process group this rank holds; returns per-attempt timings."""
    import torch
    import torch.distributed as dist
    from torch.distributed import distributed_c10d as c10d

    out = {"attempts": []}
    if not dist.is_initialized():
        out["error"] = "torch.distributed not initialized"
        return out
    mode = os.environ.get("FT_ABORT_MODE", "all")   # all | sp | world | exit | coexist
    if mode == "coexist":
        # B3-1: do NOT touch the communicator; measure whether this process's GPU still executes
        # independent work at normal speed while the main thread is stuck in the collective.
        res = []
        log("[ft-detect] coexist: main thread stack at deadline:\n" + _main_thread_stack())
        for i, reuse in enumerate((True, True, False)):   # reused buffers first, then a fresh allocation
            log(f"[ft-detect] coexist run {i} reuse={reuse}")
            res.append(_side_stream_bench(log=lambda m: log(f"[ft-detect] {m}"), reuse=reuse))
            log(f"[ft-detect] coexist run {i} result {res[-1]}")
            time.sleep(1.0)
        out["attempts"].append({"group": "coexist", "how": "side-stream matmul while stuck", "ok": True,
                                "bench_during_hang": res, "t0_ns": now_ns(), "t_ns": now_ns()})
        return out
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
            res = _abort_all_groups(lambda m: print(m, flush=True))
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


# ---- Test 3: in-process failure containment + SP2->SP1 continuation (FT_CONTAIN=1) ------------
# Collectives issued through torch.distributed are wrapped so the CPU never issues a dependent
# kernel behind an unfinished collective: async_op + poll work.is_completed() under a deadline
# (Test 1: ~+2%). A deadline miss raises PeerFailure. The denoising-step wrapper catches it,
# aborts the process group (Test 2: ~0.6 s), reconfigures the live SP group to world size 1,
# disables the model's internal sequence shard for this request, drops the dead peer from the
# scheduler's fan-out, kills the peer process, and re-runs the same step from the unchanged
# boundary state. Measurement prototype: fixed deadline, no health/unroutable signalling.
class PeerFailure(RuntimeError):
    def __init__(self, msg, group=None):
        super().__init__(msg)
        self.group = group


_CONTAIN = {"on": False, "deadline_s": 5.0, "init_deadline_s": 900.0, "failed_over": False, "injected": False, "orig": {}}


def _contain_deadline() -> float:
    # while no request is in flight (model load, warm-up barriers) allow long waits
    return _CONTAIN["deadline_s"] if _PROGRESS["in_request"] else _CONTAIN["init_deadline_s"]


def _make_contained(name, orig):
    def contained(*args, **kwargs):
        if kwargs.get("async_op") or _CONTAIN["failed_over"]:
            return orig(*args, **kwargs)
        kwargs["async_op"] = True
        work = orig(*args, **kwargs)
        if work is None:
            return None
        deadline = _contain_deadline()
        t0 = time.perf_counter()
        while not work.is_completed():
            if time.perf_counter() - t0 > deadline:
                raise PeerFailure(f"{name} not complete after {deadline:.1f}s", group=kwargs.get("group"))
        work.wait()
        return None
    contained.__name__ = f"contained_{name}"
    return contained


def _install_containment(steps):
    import torch.distributed as dist

    _CONTAIN["on"] = True
    _CONTAIN["deadline_s"] = float(os.environ.get("FT_CONTAIN_DEADLINE_S", "5"))
    for name in ("all_to_all_single", "all_gather_into_tensor", "all_reduce", "all_gather", "broadcast"):
        orig = getattr(dist, name, None)
        if orig is None or name in _CONTAIN["orig"]:
            continue
        _CONTAIN["orig"][name] = orig
        setattr(dist, name, _make_contained(name, orig))
    # the diffusion runtime bound an alias to all_gather_into_tensor at import time
    try:
        from sglang.multimodal_gen.runtime.distributed import utils as dutils
        from sglang.multimodal_gen.runtime.distributed.device_communicators import base_device_communicator as bdc

        c = _make_contained("all_gather_single", _CONTAIN["orig"].get("all_gather_into_tensor") or dutils.all_gather_single)
        dutils.all_gather_single = c
        bdc.all_gather_single = c
        steps.write({"ev": "contain", "phase": "installed", "deadline_s": _CONTAIN["deadline_s"], "patched": list(_CONTAIN["orig"]) + ["all_gather_single"]})
        print(f"[ft-detect] FT_CONTAIN armed: deadline {_CONTAIN['deadline_s']}s on {list(_CONTAIN['orig'])}", flush=True)
    except Exception as e:
        steps.write({"ev": "contain", "phase": "install_error", "error": repr(e)[:300]})


def _bounded(fn, limit_s):
    """Run fn in a daemon thread; (finished, seconds, error). A hang is a result, not a stuck rank."""
    res, done = {}, threading.Event()

    def run():
        t = time.perf_counter()
        try:
            fn()
        except Exception as e:
            res["err"] = repr(e)[:200]
        res["s"] = time.perf_counter() - t
        done.set()

    threading.Thread(target=run, daemon=True, name="ft-abort").start()
    return (True, res["s"], res.get("err")) if done.wait(limit_s) else (False, limit_s, f"hung > {limit_s:.0f} s")


def _fence_peers(rank: int, members=None) -> list:
    """Kill this node's ranks that are not in `members` (all other ranks when members is None)."""
    import signal as _sig

    killed = []
    pids = json.load(open(run_dir() / "pids.json"))
    for r, pid in pids.get("ranks", {}).items():
        if int(r) != rank and (members is None or int(r) not in members):
            try:
                os.kill(int(pid), _sig.SIGCONT); os.kill(int(pid), _sig.SIGKILL); killed.append(int(pid))
            except ProcessLookupError:
                pass
    return killed


def _sharded_components(server_args) -> list:
    """Loaded components whose weights are split across ranks: they cannot shrink to one rank in place."""
    import gc

    out = []
    import torch

    for obj in gc.get_objects():
        if not isinstance(obj, torch.nn.Module):
            continue
        grp = obj.__dict__.get("_encoder_tp_group")
        if grp is not None and getattr(grp, "world_size", 1) > 1:
            out.append(f"{type(obj).__name__}:fold={grp.world_size}")
    if server_args.tp_size > 1:
        out.append(f"dit:tp={server_args.tp_size}")
    return sorted(set(out))


def _reset_cached_parallel_state(dit_modules=(), dit_degree: int = 1, dit_rank: int = 0) -> dict:
    """Modules and caches that captured the parallel degree at construction or first use.

    Wan's DiT caches sp_size; its VAE decoder, attention blocks and spatial-parallel convs cache the
    decode world size and rank and split the latent height by them, so after the shrink they would
    decode only this rank's half. Activation-parallel state like this is recoverable in place.
    """
    import gc
    import sys as _sys

    import torch

    out = {"world_size_reset": 0, "sp_size_reset": 0, "classes": set(), "caches_cleared": []}
    dit_ids = {id(sub) for m in dit_modules if m is not None for sub in m.modules()}
    for obj in gc.get_objects():
        if not isinstance(obj, torch.nn.Module):
            continue
        in_dit = id(obj) in dit_ids
        degree, new_rank = (dit_degree, dit_rank) if in_dit else (1, 0)
        d = obj.__dict__
        ws = d.get("world_size")
        if type(ws) is int and ws > 1 and ws != degree:
            obj.world_size = degree
            if type(d.get("rank")) is int:
                obj.rank = new_rank
            out["world_size_reset"] += 1; out["classes"].add(type(obj).__name__)
        sp = d.get("sp_size")
        if type(sp) is int and sp > 1 and sp != degree:
            obj.sp_size = degree; out["sp_size_reset"] += 1; out["classes"].add(type(obj).__name__)
    # spatial-parallel convs fall back to a plain padded conv when this context var is set
    try:
        from sglang.multimodal_gen.runtime.layers import parallel_conv as pconv

        pconv._SPATIAL_PARALLEL_DECODE_DISABLED.set(True)
        for mod in list(_sys.modules.values()):
            if mod is not None and mod.__name__.startswith("sglang.multimodal_gen") and callable(getattr(mod, "spatial_parallel_decode_disabled", None)):
                mod.spatial_parallel_decode_disabled = lambda: True
        out["spatial_parallel_decode"] = "disabled"
    except Exception as e:
        out["spatial_parallel_decode"] = f"error {type(e).__name__}"
    # lru caches keyed on shape or group identity, not on the degree they were computed under
    for modname, attr in (("sglang.multimodal_gen.runtime.models.vaes.common", "_cached_decode_parallel_world_size"),):
        try:
            getattr(_sys.modules[modname], attr).cache_clear(); out["caches_cleared"].append(attr)
        except Exception:
            pass
    try:
        from sglang.multimodal_gen.runtime.layers.rotary_embedding import mrope

        for cls in vars(mrope).values():
            if isinstance(cls, type):
                for name in ("_forward_cached_from_grid", "_forward_cached"):
                    fn = cls.__dict__.get(name)
                    if fn is not None and hasattr(fn, "cache_clear"):
                        fn.cache_clear(); out["caches_cleared"].append(f"{cls.__name__}.{name}")
    except Exception:
        pass
    out["classes"] = sorted(out["classes"])
    return out


def _failover_to_sp1(steps, stage, batch, rank: int, stuck_group=None, server_args=None) -> dict:
    """Fence the peer, abort the stuck communicator, and reconfigure this process to run alone.

    FT_CONTAIN_ABORT: stuck (default; only the group whose collective missed the deadline) |
    world (_abort_process_group(), every group) | none (abandon the stuck collective).
    FT_CONTAIN_FENCE_FIRST=1 (default) kills the peer before the abort.
    """
    import faulthandler
    import gc
    import sys as _sys

    import torch
    import torch.distributed as dist
    from torch.distributed import distributed_c10d as c10d

    def mark(what):
        print(f"[ft-detect] FT_CONTAIN failover: {what} at +{(time.perf_counter() - t_start) * 1e3:.0f} ms", flush=True)

    t_start = time.perf_counter()
    if server_args is not None:
        try:
            sharded = _sharded_components(server_args)
        except Exception as e:
            sharded = [f"check_error:{type(e).__name__}"]
        if sharded:
            print(f"[ft-detect] FT_CONTAIN WARNING: weight-sharded components cannot run on one rank: {sharded}", flush=True)
    else:
        sharded = None
    tl = {"sharded_components": sharded, "abort_mode": os.environ.get("FT_CONTAIN_ABORT", "stuck"),
          "fence_first": os.environ.get("FT_CONTAIN_FENCE_FIRST", "1") == "1"}
    if tl["fence_first"]:
        tl["peers_killed"] = _fence_peers(rank); mark(f"fenced {tl['peers_killed']}")
    limit = float(os.environ.get("FT_CONTAIN_ABORT_TIMEOUT_S", "10"))
    if tl["abort_mode"] == "world":
        ok, secs, err = _bounded(lambda: c10d._abort_process_group(), limit)
    elif tl["abort_mode"] == "stuck":
        target = stuck_group if stuck_group is not None else c10d.GroupMember.WORLD
        ok, secs, err = _bounded(lambda: c10d._abort_process_group(target), limit)
    else:
        ok, secs, err = True, 0.0, None
    tl.update({"abort_finished": ok, "t_abort_s": secs, "abort_error": err})
    mark(f"abort {tl['abort_mode']} finished={ok} ({secs * 1e3:.0f} ms) {err or ''}")
    if not ok:
        faulthandler.dump_traceback(file=_sys.stderr, all_threads=True)
    if not tl["fence_first"]:
        tl["peers_killed"] = _fence_peers(rank); mark(f"fenced {tl['peers_killed']}")
    t = time.perf_counter()
    from sglang.multimodal_gen.runtime.distributed import parallel_state as ps
    from sglang.multimodal_gen.runtime.distributed.group_coordinator import GroupCoordinator

    # every coordinator (world, sp, replica, vae-decode, encoder-dp, ...) now spans this rank only;
    # GroupCoordinator ops short-circuit at world_size == 1, so no collective is issued any more.
    # Global .rank is kept: the executor's "main rank" test (rank == 0) must stay true here.
    shrunk = []
    for name, obj in vars(ps).items():
        if isinstance(obj, GroupCoordinator) and obj.world_size > 1:
            obj.world_size = 1; obj.rank_in_group = 0; obj.ranks = [obj.rank]
            for attr, val in (("ulysses_world_size", 1), ("ulysses_rank", 0), ("ring_world_size", 1), ("ring_rank", 0)):
                if hasattr(obj, attr):
                    setattr(obj, attr, val)
            shrunk.append(name)
    tl["coordinators_shrunk"] = shrunk
    reset = _reset_cached_parallel_state()
    tl["modules_sp_size_reset"] = reset["sp_size_reset"]; tl["modules_world_size_reset"] = reset["world_size_reset"]
    tl["reset_classes"] = reset["classes"]; tl["caches_cleared"] = reset["caches_cleared"]
    tl["spatial_parallel_decode"] = reset.get("spatial_parallel_decode")
    mark(f"reset cached parallel state: {reset['world_size_reset']} world_size, {reset['sp_size_reset']} sp_size in {reset['classes']}")
    try:
        batch.enable_sequence_shard = False
    except Exception as e:
        tl["batch_flag_error"] = repr(e)[:120]
    # inter-stage sync and request fan-out would wait on the dead peer forever (gloo has no abort)
    dist.barrier = lambda *a, **k: None
    n_bp = 0
    for modname, mod in list(_sys.modules.items()):
        if modname.startswith("sglang.multimodal_gen") and callable(getattr(mod, "broadcast_pyobj", None)):
            mod.broadcast_pyobj = lambda obj, *a, **k: obj; n_bp += 1
    tl["broadcast_pyobj_patched_modules"] = n_bp
    for obj in gc.get_objects():
        if type(obj).__name__ == "Scheduler" and hasattr(obj, "server_args"):
            try:
                obj.server_args.sp_degree = 1
                if hasattr(obj, "task_pipes_to_slaves"):
                    obj.task_pipes_to_slaves = []; obj.result_pipes_from_slaves = []
                tl["scheduler_patched"] = True
            except Exception as e:
                tl["scheduler_patch_error"] = repr(e)[:120]
    tl["t_reconfigure_s"] = time.perf_counter() - t
    mark("reconfigured to SP=1")
    tl["cuda_mem_alloc_MiB"] = torch.cuda.memory_allocated() / 2**20 if torch.cuda.is_available() else None
    _CONTAIN["failed_over"] = True
    return tl


def _agree_members(rank: int, world: int, settle_s: float, epoch: int) -> list:
    """Survivors register in the default store; global rank 0 (the store host) decides the set."""
    from torch.distributed import distributed_c10d as c10d

    store = c10d._get_default_store()
    store.set(f"ft/shrink/e{epoch}/alive/{rank}", "1")
    if rank == 0:
        time.sleep(settle_s)
        members = [r for r in range(world) if store.check([f"ft/shrink/e{epoch}/alive/{r}"])]
        store.set(f"ft/shrink/e{epoch}/members", ",".join(map(str, members)))
    store.wait([f"ft/shrink/e{epoch}/members"])
    return [int(v) for v in store.get(f"ft/shrink/e{epoch}/members").decode().split(",")]


def _survivor_groups(members):
    """NCCL and gloo groups of the survivors only.

    With a device-bound default group torch builds NCCL subgroups by splitting the default
    communicator, which needs every rank of it including the dead one; unbind while creating.
    """
    import torch.distributed as dist
    from torch.distributed import distributed_c10d as c10d

    dpg = c10d._get_default_group()
    saved = dpg.bound_device_id
    dpg.bound_device_id = None
    try:
        dev_group = dist.new_group(members, backend="nccl", use_local_synchronization=True)
        cpu_group = dist.new_group(members, backend="gloo", use_local_synchronization=True)
    finally:
        dpg.bound_device_id = saved
    return dev_group, cpu_group


def _rewire_coordinators(members, dev_group, cpu_group, dead) -> dict:
    """Point every coordinator that contained a dead rank at the survivor groups.

    The VAE decode group becomes a singleton: spatial-parallel decode is switched off after a
    shrink, so each rank decodes on its own.
    """
    from sglang.multimodal_gen.runtime.distributed import parallel_state as ps
    from sglang.multimodal_gen.runtime.distributed.device_communicators.cuda_communicator import CudaCommunicator
    from sglang.multimodal_gen.runtime.distributed.group_coordinator import GroupCoordinator, SequenceParallelGroupCoordinator

    done, out = set(), {"rewired": [], "singleton": []}
    for name, coord in vars(ps).items():
        if not isinstance(coord, GroupCoordinator) or id(coord) in done or not (set(coord.ranks) & dead):
            continue
        done.add(id(coord))
        if name == "_VAE_DECODE":
            coord.ranks, coord.world_size, coord.rank_in_group = [coord.rank], 1, 0
            coord.device_communicator = None
            out["singleton"].append(name)
            continue
        coord.ranks = [r for r in coord.ranks if r in members]
        coord.world_size, coord.rank_in_group = len(coord.ranks), coord.ranks.index(coord.rank)
        coord.device_group, coord.cpu_group = dev_group, cpu_group
        coord.mq_broadcaster = None
        coord.srt_custom_allreduce = None
        coord.device_communicator = CudaCommunicator(cpu_group=cpu_group, device=coord.device, device_group=dev_group,
                                                     unique_name=coord.unique_name)
        if isinstance(coord, SequenceParallelGroupCoordinator):
            if coord.ring_world_size != 1:
                raise RuntimeError("shrink supports pure Ulysses SP only (ring degree 1)")
            coord.ulysses_group = dev_group
            coord.ulysses_world_size, coord.ulysses_rank = coord.world_size, coord.rank_in_group
        out["rewired"].append(name)
    return out


def _rebind_cached_groups(world_cpu_group) -> dict:
    """Objects that copied a group handle at construction, and torch's default-group barrier."""
    import gc

    import torch.distributed as dist

    from sglang.multimodal_gen.runtime.distributed import parallel_state as ps

    n = 0
    for obj in gc.get_objects():
        if type(obj).__name__ == "GPUWorker" and "sp_cpu_group" in obj.__dict__:
            obj.sp_cpu_group = ps.get_sp_group().cpu_group; n += 1
    orig_barrier = _CONTAIN["orig"].setdefault("barrier", dist.barrier)

    def survivor_barrier(group=None, async_op=False, device_ids=None):
        if group is None:      # the default group still contains the dead rank
            return orig_barrier(group=world_cpu_group, async_op=async_op)
        return orig_barrier(group=group, async_op=async_op, device_ids=device_ids)

    dist.barrier = survivor_barrier
    return {"gpu_workers_rebound": n}


def _failover_shrink(steps, stage, batch, rank: int, stuck_group=None, server_args=None) -> dict:
    """SP=N -> SP=N-1 on the survivors: targeted abort, membership, survivor groups, rewiring."""
    import torch
    from torch.distributed import distributed_c10d as c10d

    from sglang.multimodal_gen.runtime.distributed import parallel_state as ps

    t_start = time.perf_counter()

    def mark(what):
        print(f"[ft-detect] FT_CONTAIN shrink: {what} at +{(time.perf_counter() - t_start) * 1e3:.0f} ms", flush=True)

    world = ps.get_world_group().world_size
    _CONTAIN["epoch"] = _CONTAIN.get("epoch", 0) + 1
    tl = {"mode": "shrink", "epoch": _CONTAIN["epoch"], "world_before": world}
    target = stuck_group if stuck_group is not None else c10d.GroupMember.WORLD
    ok, secs, err = _bounded(lambda: c10d._abort_process_group(target), float(os.environ.get("FT_CONTAIN_ABORT_TIMEOUT_S", "10")))
    tl.update({"abort_finished": ok, "t_abort_s": secs, "abort_error": err}); mark(f"abort finished={ok} ({secs * 1e3:.0f} ms)")
    t = time.perf_counter()
    members = _agree_members(rank, world, float(os.environ.get("FT_SHRINK_SETTLE_S", "1.0")), _CONTAIN["epoch"])
    dead = set(range(world)) - set(members)
    tl.update({"members": members, "dead": sorted(dead), "t_membership_s": time.perf_counter() - t}); mark(f"members {members}")
    if 0 in dead:
        raise RuntimeError("rank 0 (store host and request ingress) failed; shrink cannot proceed")
    tl["peers_killed"] = _fence_peers(rank, members=set(members)); mark(f"fenced {tl['peers_killed']}")
    t = time.perf_counter()
    dev_group, cpu_group = _survivor_groups(members)
    tl["t_groups_s"] = time.perf_counter() - t; mark("survivor groups built")
    t = time.perf_counter()
    tl.update(_rewire_coordinators(set(members), dev_group, cpu_group, dead))
    tl.update(_rebind_cached_groups(cpu_group))
    n, r = len(members), members.index(rank)
    reset = _reset_cached_parallel_state(dit_modules=(stage.transformer, stage.transformer_2), dit_degree=n, dit_rank=r)
    tl.update({"sp_after": ps.get_sp_group().world_size, "reset_classes": reset["classes"],
               "caches_cleared": reset["caches_cleared"], "spatial_parallel_decode": reset.get("spatial_parallel_decode"),
               "t_rewire_s": time.perf_counter() - t,
               "cuda_mem_alloc_MiB": torch.cuda.memory_allocated() / 2**20})
    mark(f"rewired to SP={tl['sp_after']} ({tl['rewired']}, singleton {tl['singleton']})")
    return tl


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
        torch.cuda.current_stream().synchronize()
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
        fail_step = int(os.environ["FT_FAIL_STEP"]) if os.environ.get("FT_FAIL_STEP") else None
        fail_rank = int(os.environ.get("FT_FAIL_RANK", "1"))
        fail_req = int(os.environ.get("FT_FAIL_REQ", "1"))
        fail_shape = os.environ.get("FT_FAIL_SHAPE")   # WxH[xF]: only requests of this shape count
        req_counter = {"n": 0}

        def _counts(batch) -> bool:
            if not fail_shape:
                return True
            d = _shape_of(batch)
            got = f"{d.get('w')}x{d.get('h')}" + (f"x{d.get('nf')}" if (d.get("nf") or 1) > 1 else "")
            return got == fail_shape

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
            counted = fail_step is not None and not getattr(ctx, "is_warmup", False) and _counts(batch)
            if counted and si == 0:
                req_counter["n"] += 1
                steps.write({"ev": "fail_req_count", "n": req_counter["n"], "target": fail_req, **_shape_of(batch)})
            if (counted and si == fail_step and rank == fail_rank and req_counter["n"] == fail_req
                    and not _CONTAIN["injected"]):
                # the failure: this rank freezes at the start of step k (SIGSTOP, as inject.py case B)
                _CONTAIN["injected"] = True
                steps.write({"ev": "fail_inject", "i": si, "t_ns": now_ns(), "pid": os.getpid(), "signal": "SIGSTOP"})
                print(f"[ft-detect] FT_FAIL_STEP: rank {rank} stopping itself at step {si}", flush=True)
                os.kill(os.getpid(), signal.SIGSTOP)
            t0 = now_ns()
            try:
                r = orig_step(self, ctx, step, batch, server_args, *a, **k)
            except PeerFailure as e:
                t_fail = now_ns()
                print(f"[ft-detect] FT_CONTAIN deadline miss at step {si}: {e}; failing over to SP1 in-process", flush=True)
                steps.write({"ev": "contain", "phase": "deadline_miss", "i": si, "t0_ns": t0, "t_ns": t_fail, "error": str(e)[:200]})
                try:
                    from sglang.multimodal_gen.runtime.distributed import parallel_state as _ps

                    if _ps.get_sp_group().world_size > 2 and os.environ.get("FT_CONTAIN_SHRINK", "1") == "1":
                        tl = _failover_shrink(steps, self, batch, rank, stuck_group=e.group, server_args=server_args)
                    else:
                        tl = _failover_to_sp1(steps, self, batch, rank, stuck_group=e.group, server_args=server_args)
                except BaseException as fe:
                    steps.write({"ev": "contain", "phase": "failover_error", "i": si, "t_ns": now_ns(), "error": repr(fe)[:300]})
                    raise
                t_sw = now_ns()
                steps.write({"ev": "contain", "phase": "failed_over", "i": si, "t_ns": t_sw, **tl})
                print(f"[ft-detect] FT_CONTAIN failed over in {(t_sw - t_fail) / 1e6:.0f} ms (abort {tl.get('t_abort_s', 0) * 1e3:.0f} ms); "
                      f"re-running step {si} at SP{tl.get('sp_after', 1)}", flush=True)
                r = orig_step(self, ctx, step, batch, server_args, *a, **k)
                steps.write({"ev": "contain", "phase": "step_recomputed", "i": si, "t0_ns": t_sw, "t_ns": now_ns()})
                print(f"[ft-detect] FT_CONTAIN step {si} recomputed at SP{tl.get('sp_after', 1)} in {(now_ns() - t_sw) / 1e6:.0f} ms", flush=True)
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
    if os.environ.get("FT_CONTAIN", "0") == "1":
        _install_containment(JsonlWriter(logs_dir() / f"steps_rank{rank}.jsonl"))
    if os.environ.get("FT_ABORT_PROBE", "0") == "1":
        if os.environ.get("FT_ABORT_MODE") == "coexist":
            def _baseline(w):
                time.sleep(float(os.environ.get("FT_COEXIST_BASELINE_DELAY_S", "90")))  # after model load/warmup
                try:
                    w.write({"ev": "coexist_baseline", "bench": [_side_stream_bench(reuse=(i > 0)) for i in range(3)]})
                except Exception as e:
                    w.write({"ev": "coexist_baseline", "error": repr(e)})
            _threading.Thread(target=_baseline, args=(JsonlWriter(logs_dir() / f"steps_rank{rank}.jsonl"),), daemon=True).start()
        _threading.Thread(target=_abort_probe_thread, args=(JsonlWriter(logs_dir() / f"steps_rank{rank}.jsonl"), rank),
                          daemon=True, name="ft-abort-probe").start()
        print(f"[ft-detect] FT_ABORT_PROBE armed on rank{rank}: deadline {os.environ.get('FT_ABORT_DEADLINE_S', '10')}s, "
              f"mode {os.environ.get('FT_ABORT_MODE', 'all')}", flush=True)

    from sglang.multimodal_gen.runtime.managers.gpu_worker import run_scheduler_process

    return run_scheduler_process(local_rank, rank, master_port, server_args, *args, **kwargs)


def _pids_writer(num_gpus: int, nnodes: int = 1, node_rank: int = 0) -> None:
    import psutil

    local = num_gpus // nnodes
    local_ranks = range(node_rank * local, node_rank * local + local)
    deadline = time.time() + 7200
    while time.time() < deadline:
        ranks = {}
        for i in local_ranks:
            p = run_dir() / "pids" / f"rank{i}.pid"
            if p.exists():
                try:
                    ranks[str(i)] = int(p.read_text().strip())
                except ValueError:
                    pass
        if len(ranks) == local:
            me = psutil.Process(os.getpid())
            others = {}
            for c in me.children(recursive=False):
                if c.pid in ranks.values():
                    continue
                try:
                    others[str(c.pid)] = c.name() + " " + " ".join(c.cmdline()[-2:])
                except Exception:
                    others[str(c.pid)] = "?"
            rec = {"http_server": os.getpid(), "tp_size": num_gpus, "nnodes": nnodes, "node_rank": node_rank,
                   "stack": "sglang-diffusion", "ranks": ranks,
                   "detokenizer": None, "other_children": others, "t_ready_ns": now_ns(), "wall_ready": wall(),
                   "argv": sys.argv[1:]}
            with open(run_dir() / "pids.json", "w") as f:
                json.dump(rec, f, indent=2)
            print(f"[ft-detect] wrote {run_dir() / 'pids.json'}: {rec}", flush=True)
            return
        time.sleep(1)


def _patch_local_gpu_memory_probe() -> None:
    """Multi-node: SGLang 0.5.19's auto-tuner probes device ids base_gpu_id..base_gpu_id+num_gpus,
    but num_gpus counts every node, so a 2-GPU node asks for devices 2 and 3 and fails at startup.
    Probe this node's GPUs only; single-node behaviour is unchanged.
    """
    from sglang.multimodal_gen.runtime.platforms import current_platform
    from sglang.multimodal_gen.runtime.server_args import auto_tune

    def local_min_available_gb(self):
        args = self.server_args
        if current_platform.is_cpu():
            return None
        local = max(1, args.num_gpus // max(1, args.nnodes))
        return min(current_platform.get_available_gpu_memory(device_id=d, empty_cache=False)
                   for d in range(args.base_gpu_id, args.base_gpu_id + local))

    auto_tune.ServerArgsAutoTuner._get_min_available_device_memory_gb = local_min_available_gb


def main() -> None:
    import sglang.multimodal_gen.runtime.launch_server as ls
    from sglang.multimodal_gen.runtime.server_args.server_args import prepare_server_args

    ensure_run_dirs()
    _patch_local_gpu_memory_probe()
    server_args = prepare_server_args(sys.argv[1:])
    num_gpus = int(getattr(server_args, "num_gpus", 1))
    with open(run_dir() / "launch.json", "w") as f:
        json.dump({"t_launch_ns": now_ns(), "wall_launch": wall(), "pid": os.getpid(), "argv": sys.argv[1:],
                   "stack": "sglang-diffusion"}, f)

    # launch_server() reads this name from its module globals when it spawns workers.
    assert hasattr(ls, "run_scheduler_process"), "launch_server module has no run_scheduler_process global; version drift"
    ls.run_scheduler_process = run_scheduler_process_wrapped
    threading.Thread(target=_pids_writer, args=(num_gpus, int(server_args.nnodes), int(server_args.node_rank)), daemon=True).start()

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
