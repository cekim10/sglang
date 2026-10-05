"""Why does _abort_process_group return in 0.6 s in the option-3 probe but hang in the server?

The option-3 probe used one lazily created NCCL group. SGLang Diffusion initialises the default
group eagerly with a bound device (device_id), so every new_group() is an ncclCommSplit of the
default communicator, and it adds gloo CPU groups next to each NCCL group. This probe rebuilds
that layout and tries one abort variant per run. Rank 1 SIGSTOPs itself; rank 0's all-to-all on
the stuck group can never complete; nothing else is issued behind it (contained discipline).

  --init   lazy | eager       eager = init_process_group(device_id=...) as SGLang does
  --stuck  default | sub      which group the stuck all-to-all runs on
  --abort  world | stuck | none
             world = _abort_process_group() (what Test 3 called); stuck = only the stuck group;
             none  = abandon the stuck collective without aborting
  --gloo   1 adds gloo groups like GroupCoordinator's cpu_group
  --fence  1 SIGKILLs rank 1 before the abort (fence first, then abort)

Records, each bounded so a hang is a result and not a stuck run: abort wall time (or
"hung after N s" plus all thread stacks), whether a launch on the main stream completes after
the abort, and whether torch.cuda.synchronize() (device-wide, waits on NCCL streams too) returns.

    ./run_abort_matrix.sh      # every case, results/abort_matrix/summary.md
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
    """Run fn in a daemon thread; return (ok, ms or None, error)."""
    res = {}
    done = threading.Event()

    def run():
        t = time.perf_counter()
        try:
            fn()
            res["ms"] = (time.perf_counter() - t) * 1e3
        except Exception as e:
            res["err"] = repr(e)[:200]
            res["ms"] = (time.perf_counter() - t) * 1e3
        done.set()

    threading.Thread(target=run, daemon=True).start()
    if not done.wait(limit_s):
        return False, None, f"hung > {limit_s:.0f} s"
    return "err" not in res, round(res["ms"], 1), res.get("err")


def all_stacks() -> str:
    r, w = os.pipe()
    faulthandler.dump_traceback(file=w, all_threads=True)
    os.close(w)
    with os.fdopen(r) as f:
        return f.read()[-6000:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--init", choices=["lazy", "eager"], default="eager")
    ap.add_argument("--stuck", choices=["default", "sub"], default="sub")
    ap.add_argument("--abort", choices=["world", "stuck", "none"], default="world")
    ap.add_argument("--gloo", type=int, default=1)
    ap.add_argument("--fence", type=int, default=0)
    ap.add_argument("--deadline-s", type=float, default=3.0)
    ap.add_argument("--abort-limit-s", type=float, default=20.0)
    ap.add_argument("--numel", type=int, default=16 * 1024 * 1024)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    case = {"init": a.init, "stuck": a.stuck, "abort": a.abort, "gloo": a.gloo, "fence": a.fence}
    res = {"case": case, "rank": a.rank}

    def dump(exit_kind):
        res["exit"] = exit_kind; res["log"] = LOG
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)

    def watchdog():
        time.sleep(a.deadline_s + 2 * a.abort_limit_s + 40)
        log("watchdog.exit"); res["stacks_at_watchdog"] = all_stacks(); dump("watchdog"); os._exit(2)

    threading.Thread(target=watchdog, daemon=True).start()
    torch.cuda.set_device(0)
    dev = torch.device("cuda", 0)
    if a.init == "eager":
        dist.init_process_group("nccl", rank=a.rank, world_size=2, device_id=dev)
    else:
        dist.init_process_group("nccl", rank=a.rank, world_size=2)
    sub = dist.new_group([0, 1], backend="nccl")
    extra = []
    if a.gloo:
        extra = [dist.new_group([0, 1], backend="gloo"), dist.new_group([0, 1], backend="gloo")]
    grp = None if a.stuck == "default" else sub
    x = torch.randn(a.numel, device=dev, dtype=torch.bfloat16); y = torch.empty_like(x)
    for _ in range(3):
        dist.all_to_all_single(y, x); dist.all_to_all_single(y, x, group=sub)
    for g in extra:
        dist.barrier(group=g)
    torch.cuda.synchronize()
    store = dist.distributed_c10d._get_default_store()
    if a.rank == 1:
        store.set("rank1_pid", str(os.getpid()))
        dist.barrier(group=extra[0]) if extra else dist.barrier()
        log("rank1.sigstop_self"); dump("stopped")
        os.kill(os.getpid(), signal.SIGSTOP)
        time.sleep(1); dump("resumed"); return 0
    dist.barrier(group=extra[0]) if extra else dist.barrier()
    peer = int(store.get("rank1_pid").decode())
    time.sleep(0.5)
    m = torch.randn(2048, 2048, device=dev, dtype=torch.bfloat16); mm = torch.empty_like(m)
    torch.cuda.synchronize()
    log("enqueue_stuck", group=a.stuck)
    work = dist.all_to_all_single(y, x, group=grp, async_op=True)
    t_dead = time.perf_counter() + a.deadline_s
    while time.perf_counter() < t_dead and not work.is_completed():
        time.sleep(0.005)
    log("deadline_miss", completed=work.is_completed())
    if a.fence:
        os.kill(peer, signal.SIGCONT); os.kill(peer, signal.SIGKILL); log("fenced_peer", pid=peer)
    if a.abort == "world":
        ok, ms, err = bounded(lambda: dist.distributed_c10d._abort_process_group(), a.abort_limit_s)
    elif a.abort == "stuck":
        target = grp if grp is not None else dist.group.WORLD
        ok, ms, err = bounded(lambda: dist.distributed_c10d._abort_process_group(target), a.abort_limit_s)
    else:
        ok, ms, err = True, 0.0, None
    res["abort"] = {"ok": ok, "ms": ms, "err": err}
    log("abort_result", **res["abort"])
    if not ok and err and err.startswith("hung"):
        res["stacks_during_abort"] = all_stacks()
    # main stream still usable?
    e1 = torch.cuda.Event()

    def launch():
        torch.matmul(m, m, out=mm); e1.record()

    lok, lms, lerr = bounded(launch, 5.0)
    done = False
    if lok:
        for _ in range(500):
            if e1.query():
                done = True; break
            time.sleep(0.01)
    res["main_stream_after"] = {"launch_returned": lok, "completed": done, "err": lerr}
    log("main_stream_after", **res["main_stream_after"])
    sok, sms, serr = bounded(torch.cuda.synchronize, 5.0)
    res["device_sync_after"] = {"ok": sok, "ms": sms, "err": serr}
    log("device_sync_after", **res["device_sync_after"])
    res["stuck_work_completed"] = work.is_completed() if a.abort == "none" else None
    dump("normal"); log("done")
    os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
