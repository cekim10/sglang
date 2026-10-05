"""Option-3 probe: does a stuck NCCL collective kill the CUDA context, or does issuing past it?

Two ranks (two GPUs, NCCL over SHM like the serving path). After a warm-up, rank 1 SIGSTOPs
itself, so rank 0's next all-to-all can never complete. Rank 0 then behaves in one of two ways:

  flood        : keep launching matmuls on the same stream behind the stuck collective, the way a
                 DiT forward does (B3-1 reproduction). Records how many launches fit before
                 cudaLaunchKernel blocks (= the context's launch-queue headroom).
  disciplined  : enqueue the collective, record an event, and poll it without issuing any further
                 work (progress-deadline style). After FT_DEADLINE_S, try the escape hatches:
                 _abort_process_group, then a launch on the main stream, then destroy_process_group.

In both modes a side thread runs an event-synchronised matmul benchmark on its own stream every
second and records whether it completes and how fast, before, during and after the hang.
Everything is written to --out as JSON; a watchdog thread hard-exits the process at --max-s so a
hang never needs torch's 600 s timeout.

Run with run_launch_discipline.sh (sets MASTER_ADDR/PORT, launches both ranks, SIGCONTs/cleans up).
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
import time

import torch
import torch.distributed as dist

T0 = time.perf_counter()
LOG = []
LOCK = threading.Lock()


def log(ev: str, **kw):
    rec = {"t": round(time.perf_counter() - T0, 4), "ev": ev, **kw}
    with LOCK:
        LOG.append(rec)
    print(f"[{rec['t']:8.3f}] {ev} {json.dumps(kw) if kw else ''}", flush=True)


def dump(path: str, extra: dict):
    with LOCK:
        data = {"log": list(LOG), **extra}
    with open(path, "w") as f:
        json.dump(data, f, indent=1)


def side_bench_loop(stop: threading.Event, size: int = 2048, n_iter: int = 20):
    """Independent work on a side stream; buffers allocated once, event-synchronised only."""
    dev = torch.device("cuda", torch.cuda.current_device())
    stream = torch.cuda.Stream(device=dev)
    with torch.cuda.stream(stream):
        a = torch.randn(size, size, device=dev, dtype=torch.bfloat16)
        b = torch.randn(size, size, device=dev, dtype=torch.bfloat16)
        c = torch.empty(size, size, device=dev, dtype=torch.bfloat16)
        d = torch.empty(64 * 1024 * 1024 // 2, device=dev, dtype=torch.bfloat16)
        torch.matmul(a, b, out=c); torch.cuda.Event().record(stream).synchronize()
    i = 0
    while not stop.is_set():
        i += 1
        with torch.cuda.stream(stream):
            e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
            log("side.launch_begin", i=i)
            t = time.perf_counter(); e0.record(stream); d.zero_()
            for _ in range(n_iter):
                torch.matmul(a, b, out=c)
            e1.record(stream); t_launch = (time.perf_counter() - t) * 1e3
            log("side.launch_done", i=i, launch_ms=round(t_launch, 2))
            t = time.perf_counter(); e1.synchronize()
            log("side.sync_done", i=i, wait_ms=round((time.perf_counter() - t) * 1e3, 1), matmul_ms=round(e0.elapsed_time(e1) / n_iter, 3))
        stop.wait(1.0)


def watchdog(max_s: float, out: str, state: dict):
    time.sleep(max_s)
    log("watchdog.exit", note="hard exit; something is still blocked", state=state)
    dump(out, {"state": state, "exit": "watchdog"})
    os._exit(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--world", type=int, default=2)
    ap.add_argument("--mode", choices=["flood", "disciplined"], default="disciplined")
    ap.add_argument("--deadline-s", type=float, default=10.0)
    ap.add_argument("--max-s", type=float, default=120.0)
    ap.add_argument("--numel", type=int, default=16 * 1024 * 1024, help="all-to-all size in bf16 elements (32 MiB default)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    state = {"mode": a.mode, "rank": a.rank, "launches_before_block": None, "abort": None}
    threading.Thread(target=watchdog, args=(a.max_s, a.out, state), daemon=True).start()

    torch.cuda.set_device(0)   # each rank sees one GPU via CUDA_VISIBLE_DEVICES
    dist.init_process_group("nccl", rank=a.rank, world_size=a.world)
    dev = torch.device("cuda", 0)
    buf_in = torch.randn(a.numel, device=dev, dtype=torch.bfloat16)
    buf_out = torch.empty_like(buf_in)
    log("init_done", nccl=".".join(map(str, torch.cuda.nccl.version())), torch=torch.__version__)

    # warm-up: both ranks participate
    for i in range(5):
        dist.all_to_all_single(buf_out, buf_in)
    torch.cuda.synchronize()
    log("warmup_done")
    dist.barrier()

    if a.rank == 1:
        # become the failed peer: stop right here, never join the next collective
        log("rank1.sigstop_self")
        dump(a.out, {"state": state, "exit": "stopped"})
        os.kill(os.getpid(), signal.SIGSTOP)
        time.sleep(1)
        log("rank1.resumed")
        dump(a.out, {"state": state, "exit": "resumed"})
        return 0

    # rank 0: start the side benchmark first (it must be running while the hang develops)
    stop = threading.Event()
    threading.Thread(target=side_bench_loop, args=(stop,), daemon=True).start()
    time.sleep(2.5)
    time.sleep(0.5)  # give rank 1 time to stop

    main_stream = torch.cuda.current_stream()
    log("main.enqueue_stuck_collective")
    work = dist.all_to_all_single(buf_out, buf_in, async_op=True)
    ev = torch.cuda.Event(); ev.record(main_stream)
    log("main.collective_enqueued")

    if a.mode == "flood":
        x = torch.randn(1024, 1024, device=dev, dtype=torch.bfloat16); y = torch.empty_like(x)
        n = 0
        slow = None
        while True:
            t = time.perf_counter()
            torch.matmul(x, x, out=y)       # same stream, behind the stuck collective
            dt = (time.perf_counter() - t) * 1e3
            n += 1
            if dt > 500:
                slow = dt; break
            if n % 500 == 0:
                log("main.flood_progress", launches=n, last_launch_ms=round(dt, 3))
        state["launches_before_block"] = n
        log("main.launch_blocked_or_slow", launches=n, launch_ms=round(slow, 1))
        # if we get here, the launch returned after a long block; keep flooding until the watchdog
        while True:
            torch.matmul(x, x, out=y)
    else:
        t_dead = time.perf_counter() + a.deadline_s
        polls = 0
        while time.perf_counter() < t_dead:
            polls += 1
            if ev.query() or work.is_completed():
                log("main.collective_completed_unexpectedly"); break
            time.sleep(0.01)
        log("main.deadline_miss", polls=polls, event_done=ev.query())
        # escape hatch 1: abort the process group from the (unblocked) main thread
        res = {}
        done = threading.Event()

        def do_abort():
            from torch.distributed import distributed_c10d as c10d
            t = time.perf_counter()
            try:
                c10d._abort_process_group()
                res["abort"] = {"ok": True, "ms": round((time.perf_counter() - t) * 1e3, 1)}
            except Exception as e:
                res["abort"] = {"ok": False, "error": repr(e)[:200], "ms": round((time.perf_counter() - t) * 1e3, 1)}
            done.set()

        log("main.abort_call")
        threading.Thread(target=do_abort, daemon=True).start()
        done.wait(30.0)
        state["abort"] = res.get("abort", {"ok": False, "error": "abort did not return within 30 s"})
        log("main.abort_result", **state["abort"])
        # escape hatch 2: can the main stream run again? (new launch + event-synchronised)
        try:
            e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
            z = torch.randn(2048, 2048, device=dev, dtype=torch.bfloat16); zz = torch.empty_like(z)
            t = time.perf_counter(); e0.record(main_stream); torch.matmul(z, z, out=zz); e1.record(main_stream)
            log("main.post_abort_launch_returned", launch_ms=round((time.perf_counter() - t) * 1e3, 2))
            ok = False
            for _ in range(500):   # 5 s of polling instead of a blocking sync
                if e1.query():
                    ok = True; break
                time.sleep(0.01)
            log("main.post_abort_main_stream", completed=ok, matmul_ms=round(e0.elapsed_time(e1), 3) if ok else None,
                stuck_event_done=ev.query(), work_completed=work.is_completed())
            state["main_stream_after_abort"] = ok
        except Exception as e:
            log("main.post_abort_error", error=repr(e)[:200])
        # escape hatch 3: destroy
        try:
            t = time.perf_counter(); dist.destroy_process_group(); log("main.destroy_done", ms=round((time.perf_counter() - t) * 1e3, 1))
        except Exception as e:
            log("main.destroy_error", error=repr(e)[:200])
        time.sleep(3.0)   # let the side benchmark record a few more iterations
        stop.set()
        dump(a.out, {"state": state, "exit": "normal"})
        log("done")
        os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
