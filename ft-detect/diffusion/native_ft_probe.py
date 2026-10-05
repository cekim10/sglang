"""Does native NCCL fault tolerance make issuance discipline unnecessary?

Two ranks (rank 0 on the first GPU, rank 1 on the second). Like SGLang Diffusion, the default group
is initialised eagerly with a bound device and the stuck collective runs on a split subgroup.
Rank 1 SIGSTOPs itself, so rank 0's all-to-all on the subgroup can never complete. Rank 0 then
issues work the way a runtime would (--issue), and a separate detector thread, after
--deadline-s, recovers with a native primitive (--recover) and checks whether the survivor is usable.

  --issue  none   async collective, nothing issued behind it (the containment discipline)
           k1|k8  synchronous collective (compute stream waits on it), then 1 or 8 kernels that
                  read its output, as the next layer of a DiT forward would
           flood  synchronous collective, then dependent kernels until a launch blocks (run-ahead)
  --recover abort         torch _abort_process_group(subgroup)   (ncclCommAbort)
            shrink_abort  dist.shrink_group([1], subgroup, SHRINK_ABORT)   (ncclCommShrink, NCCL >= 2.27)
  TORCH_NCCL_USE_COMM_NONBLOCKING=0|1 is set by the runner for both ranks.

Recorded, every step bounded so a hang is a result: whether the recovery call returns, whether the
blocked main thread gets unblocked, whether new work on the same stream completes, whether
torch.cuda.synchronize() returns (no kernel left stuck on the device), and for shrink whether a
collective on the new group completes. Survivor recoverable = all of these.

    ./run_native_ft.sh          # every case -> results/native_ft/summary.md
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import os
import signal
import sys
import threading
import time

import torch
import torch.distributed as dist

T0 = time.perf_counter()
LOG: list = []


def log(ev, **kw):
    rec = {"t": round(time.perf_counter() - T0, 3), "ev": ev, **kw}
    LOG.append(rec)
    print(f"[{rec['t']:8.3f}] {ev} {json.dumps(kw) if kw else ''}", flush=True)


def bounded(fn, limit_s):
    res, done = {}, threading.Event()

    def run():
        t = time.perf_counter()
        try:
            res["value"] = fn()
        except Exception as e:
            res["err"] = repr(e)[:300]
        res["ms"] = round((time.perf_counter() - t) * 1e3, 1)
        done.set()

    threading.Thread(target=run, daemon=True).start()
    if not done.wait(limit_s):
        return {"ok": False, "ms": None, "err": f"hung > {limit_s:.0f} s"}
    return {"ok": "err" not in res, "ms": res["ms"], "err": res.get("err"), "value": res.get("value")}


def poll_event(ev, limit_s):
    t = time.perf_counter()
    while time.perf_counter() - t < limit_s:
        if ev.query():
            return True
        time.sleep(0.01)
    return False


def stacks() -> str:
    r, w = os.pipe()
    faulthandler.dump_traceback(file=w, all_threads=True)
    os.close(w)
    with os.fdopen(r) as f:
        return f.read()[-6000:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--issue", choices=["none", "k1", "k8", "flood"], default="k1")
    ap.add_argument("--recover", choices=["abort", "shrink_abort"], default="shrink_abort")
    ap.add_argument("--deadline-s", type=float, default=3.0)
    ap.add_argument("--limit-s", type=float, default=20.0)
    ap.add_argument("--numel", type=int, default=16 * 1024 * 1024)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    nb = os.environ.get("TORCH_NCCL_USE_COMM_NONBLOCKING", "0")
    res = {"case": {"issue": a.issue, "recover": a.recover, "nonblocking": nb}, "rank": a.rank,
           "torch": torch.__version__, "nccl": ".".join(map(str, torch.cuda.nccl.version())),
           "has_shrink_group": hasattr(dist, "shrink_group")}

    def dump(kind):
        res["exit"] = kind; res["log"] = LOG
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1, default=str)

    def watchdog():
        time.sleep(a.deadline_s + 5 * a.limit_s + 30)
        res["stacks_at_watchdog"] = stacks(); log("watchdog.exit"); dump("watchdog"); os._exit(2)

    threading.Thread(target=watchdog, daemon=True).start()
    torch.cuda.set_device(0)
    dev = torch.device("cuda", 0)
    dist.init_process_group("nccl", rank=a.rank, world_size=2, device_id=dev)
    sub = dist.new_group([0, 1], backend="nccl")
    gloo = dist.new_group([0, 1], backend="gloo")
    n = 2048
    x = torch.randn(a.numel, device=dev, dtype=torch.bfloat16); y = torch.empty_like(x)
    w = torch.randn(n, n, device=dev, dtype=torch.bfloat16); c = torch.empty(n, n, device=dev, dtype=torch.bfloat16)
    for _ in range(3):
        dist.all_to_all_single(y, x, group=sub)
    torch.cuda.synchronize()
    log("init_done", torch=res["torch"], nccl=res["nccl"], has_shrink_group=res["has_shrink_group"], nonblocking=nb)
    store = dist.distributed_c10d._get_default_store()
    if a.rank == 1:
        store.set("rank1_pid", str(os.getpid()))
        dist.barrier(group=gloo)
        log("rank1.sigstop_self"); dump("stopped")
        os.kill(os.getpid(), signal.SIGSTOP)
        time.sleep(1); dump("resumed"); return 0
    dist.barrier(group=gloo)
    time.sleep(0.5)
    yv = y[: n * n].view(n, n)
    state = {"issued": 0, "main_blocked_at": None, "main_returned_after_recovery": None, "recovered_at": None}

    def detector():
        time.sleep(a.deadline_s)
        log("deadline_miss", issued=state["issued"], main_blocked=state["main_blocked_at"] is not None)
        if a.recover == "abort":
            r = bounded(lambda: dist.distributed_c10d._abort_process_group(sub), a.limit_s)
        else:
            flag = getattr(dist, "SHRINK_ABORT", getattr(dist.distributed_c10d, "SHRINK_ABORT", 1))
            r = bounded(lambda: dist.shrink_group([1], group=sub, shrink_flags=flag), a.limit_s)
        state["recovered_at"] = time.perf_counter()
        newg = r.pop("value", None)
        res["recovery"] = r; log("recovery_result", **r)
        if not r["ok"]:
            res["stacks_during_recovery"] = stacks()
        time.sleep(1.0)
        res["main_thread_unblocked"] = (state["main_blocked_at"] is None) or (state["main_returned_after_recovery"] is True)
        # new dependent-free work on the same (default) stream
        e = torch.cuda.Event()

        def launch():
            torch.matmul(w, w, out=c); e.record()

        lr = bounded(launch, 5.0)
        res["new_work_same_stream"] = {"launch_returned": lr["ok"], "completed": lr["ok"] and poll_event(e, 5.0), "err": lr["err"]}
        log("new_work_same_stream", **res["new_work_same_stream"])
        res["device_sync"] = bounded(torch.cuda.synchronize, 5.0); res["device_sync"].pop("value", None)
        log("device_sync", **res["device_sync"])
        if a.recover == "shrink_abort" and newg is not None:
            t = torch.ones(1024, device=dev)

            def ar():
                wk = dist.all_reduce(t, group=newg, async_op=True)
                t0 = time.perf_counter()
                while not wk.is_completed():
                    if time.perf_counter() - t0 > 5:
                        raise TimeoutError("all_reduce on the shrunk group did not complete")
                    time.sleep(0.01)
                return float(t[0].item())

            g = bounded(ar, 8.0)
            res["new_group_collective"] = g; log("new_group_collective", **g)
        res["survivor_recoverable"] = bool(r["ok"] and res["main_thread_unblocked"] and res["new_work_same_stream"]["completed"]
                                           and res["device_sync"]["ok"]
                                           and (a.recover != "shrink_abort" or res.get("new_group_collective", {}).get("ok")))
        log("verdict", survivor_recoverable=res["survivor_recoverable"], issued=state["issued"])
        dump("normal"); os._exit(0)

    threading.Thread(target=detector, daemon=True).start()
    log("enqueue_stuck_collective", issue=a.issue)
    if a.issue == "none":
        dist.all_to_all_single(y, x, group=sub, async_op=True)    # nothing waits on it, nothing issued behind it
        while True:
            time.sleep(1.0)
    dist.all_to_all_single(y, x, group=sub)                       # compute stream now waits on the collective
    limit = {"k1": 1, "k8": 8, "flood": 10 ** 9}[a.issue]
    while state["issued"] < limit:
        t = time.perf_counter()
        state["main_blocked_at"] = t
        torch.matmul(yv, w, out=c)                                  # reads the collective's output
        dt = time.perf_counter() - t
        if state["recovered_at"] is not None and t < state["recovered_at"]:
            state["main_returned_after_recovery"] = True
            log("main_launch_returned_after_recovery", blocked_ms=round(dt * 1e3, 1), issued=state["issued"])
        state["main_blocked_at"] = None
        state["issued"] += 1
        if state["recovered_at"] is not None:
            break
        if dt > 0.5:
            log("main_launch_slow", ms=round(dt * 1e3, 1), issued=state["issued"])
    log("issued_all", issued=state["issued"])
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    sys.exit(main())
