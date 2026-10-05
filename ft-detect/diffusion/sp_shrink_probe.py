"""SP=4 -> SP=3 in-place continuation after one rank fails (standalone; NCCL on GPUs or gloo on CPU).

Generalises Test 2 from "fall back to one rank" to "re-form the surviving distributed group":
a replicated-state Ulysses block (the Wan case) runs N steps at SP=4; the last rank SIGSTOPs at the
start of step k; the three survivors detect the stalled all-to-all with a deadline, abort only that
group, agree on membership through the default store, build a new NCCL (or gloo) group and a gloo
CPU group among themselves, and recompute step k and finish at SP=3 from the boundary state.

SGLang binds the default group to a device (eager init), and torch then creates NCCL subgroups by
splitting the default communicator, which needs every rank of it, including the dead one. The
probe reproduces that layout and builds the survivor group with the device unbound temporarily
(members-only store rendezvous, use_local_synchronization=True).

References: SP=4 (all ranks), SP=3 (a group of ranks 0-2 made before the failure), SP=1 (rank 0).
PASS = the survivors finish every remaining step at SP=3 and the result equals the SP=3 reference
(and lies within the SP=1/SP=4 control). Also measured: detection, abort, rendezvous, group
creation, first SP=3 step, and a broadcast of a Python object over the new CPU group (what
SGLang's scheduler uses to hand requests to the other ranks).

Launch one process per rank with RANK, WORLD_SIZE, LOCAL_RANK, MASTER_ADDR, MASTER_PORT set
(see run_shrink.sh). The failing rank must not be 0: rank 0 hosts the store.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import threading
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed import distributed_c10d as c10d

T0 = time.perf_counter()
LOG: list = []
RES_SCALE = 0.1
SHAPES = {  # tokens divisible by 12 and heads divisible by 4 and 3, so SP=4 and SP=3 both shard evenly
    "wan": {"tokens": 6144, "hidden": 3072, "heads": 24, "layers": 8},
    "small": {"tokens": 1536, "hidden": 768, "heads": 12, "layers": 4},
    "cpu": {"tokens": 96, "hidden": 192, "heads": 12, "layers": 4},
}


def log(ev, **kw):
    rec = {"t": round(time.perf_counter() - T0, 4), "ev": ev, **kw}
    LOG.append(rec)
    print(f"[r{os.environ.get('RANK')} {rec['t']:8.3f}] {ev} {json.dumps(kw, default=str) if kw else ''}", flush=True)


class Deadline(Exception):
    pass


class Layer:
    def __init__(self, hidden, heads, dev, dtype, gen):
        self.heads, self.dd = heads, hidden // heads
        k = lambda *s: (torch.randn(*s, generator=gen, device=dev, dtype=torch.float32) * 0.02).to(dtype)
        self.wqkv, self.wo, self.w1, self.w2 = k(hidden, 3 * hidden), k(hidden, hidden), k(hidden, 4 * hidden), k(4 * hidden, hidden)


def rms(x):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype)


def _wait(work, deadline_s, what):
    t0 = time.perf_counter()
    while not work.is_completed():
        if time.perf_counter() - t0 > deadline_s:
            raise Deadline(f"{what} not complete after {deadline_s}s")
        time.sleep(0.0005)
    work.wait()


def layer_sp(L: Layer, x, group, n, r, deadline_s):
    """Ulysses block at degree n on `group`; this rank is r in it. x is the replicated full state."""
    S, H, hd = x.shape[0], L.heads, L.dd
    s_loc = S // n
    h = rms(x)
    qkv = (h[r * s_loc:(r + 1) * s_loc] @ L.wqkv).view(s_loc, 3, H, hd)
    send = qkv.view(s_loc, 3, n, H // n, hd).permute(2, 0, 1, 3, 4).contiguous()
    recv = torch.empty_like(send)
    _wait(dist.all_to_all_single(recv, send, group=group, async_op=True), deadline_s, "all-to-all")
    full = recv.permute(2, 0, 1, 3, 4).reshape(3, S, H // n, hd)
    q, k, v = (full[i].transpose(0, 1).unsqueeze(0) for i in range(3))
    o = F.scaled_dot_product_attention(q, k, v).squeeze(0).transpose(0, 1).contiguous()
    send2 = o.view(n, s_loc, H // n, hd).contiguous()
    recv2 = torch.empty_like(send2)
    _wait(dist.all_to_all_single(recv2, send2, group=group, async_op=True), deadline_s, "all-to-all")
    attn_local = recv2.permute(1, 0, 2, 3).reshape(s_loc, H * hd)
    gathered = [torch.empty_like(attn_local) for _ in range(n)]
    _wait(dist.all_gather(gathered, attn_local, group=group, async_op=True), deadline_s, "all-gather")
    x = x + RES_SCALE * (torch.cat(gathered, 0) @ L.wo)
    return x + RES_SCALE * (F.silu(rms(x) @ L.w1) @ L.w2)


def layer_sp1(L: Layer, x):
    S, H, hd = x.shape[0], L.heads, L.dd
    qkv = (rms(x) @ L.wqkv).view(S, 3, H, hd)
    q, k, v = (qkv[:, i].transpose(0, 1).unsqueeze(0) for i in range(3))
    o = F.scaled_dot_product_attention(q, k, v).squeeze(0).transpose(0, 1).reshape(S, H * hd)
    x = x + RES_SCALE * (o @ L.wo)
    return x + RES_SCALE * (F.silu(rms(x) @ L.w1) @ L.w2)


def cmp(a, b):
    d = (a.float() - b.float()).abs()
    return {"exact": bool(torch.equal(a, b)), "max_abs": float(d.max()), "rel_l2": float(d.norm() / (b.float().norm() + 1e-12))}


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.current_stream().synchronize()


def survivor_group(members, backend, dev):
    """New groups among the survivors only; the dead rank is still in the default communicator."""
    dpg = c10d._get_default_group()
    saved = getattr(dpg, "bound_device_id", None)
    unbound = False
    if saved is not None:
        try:
            dpg.bound_device_id = None      # stop torch from ncclCommSplit-ing the default communicator
            unbound = True
        except Exception as e:
            log("unbind_failed", error=repr(e)[:200])
    try:
        log("survivor_group.device_group_begin", unbound=unbound)
        dev_group = dist.new_group(members, backend=backend, use_local_synchronization=True)
        log("survivor_group.device_group_created")
        cpu_group = dist.new_group(members, backend="gloo", use_local_synchronization=True)
        log("survivor_group.cpu_group_created")
    finally:
        if unbound:
            dpg.bound_device_id = saved
    return dev_group, cpu_group, unbound


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["nccl", "gloo"], default="nccl")
    ap.add_argument("--shape", choices=sorted(SHAPES), default="wan")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--fail-step", type=int, default=4)
    ap.add_argument("--deadline-s", type=float, default=5.0)
    ap.add_argument("--settle-s", type=float, default=3.0, help="leader waits this long for survivors to register")
    ap.add_argument("--max-s", type=float, default=600.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    fail_rank = world - 1
    res = {"rank": rank, "world": world, "fail_rank": fail_rank, "host": socket.gethostname(), "args": vars(a)}

    def dump(exit_kind):
        res["exit"] = exit_kind; res["log"] = LOG
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1, default=str)

    def watchdog():
        time.sleep(a.max_s); log("watchdog.exit"); dump("watchdog"); os._exit(2)

    threading.Thread(target=watchdog, daemon=True).start()
    if a.backend == "nccl":
        local = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local)
        dev, dtype = torch.device("cuda", local), torch.bfloat16
        dist.init_process_group("nccl", rank=rank, world_size=world, device_id=dev)   # eager, as SGLang does
    else:
        dev, dtype = torch.device("cpu"), torch.float32
        dist.init_process_group("gloo", rank=rank, world_size=world)
    store = c10d._get_default_store()
    store.set(f"pid/{rank}", f"{socket.gethostname()}:{os.getpid()}")
    cfg = SHAPES[a.shape]
    gen = torch.Generator(device=dev).manual_seed(1234)
    layers = [Layer(cfg["hidden"], cfg["heads"], dev, dtype, gen) for _ in range(cfg["layers"])]
    x0 = torch.randn(cfg["tokens"], cfg["hidden"], generator=gen, device=dev, dtype=torch.float32).to(dtype)
    # gloo between machines must work before the failure (it carries the scheduler's request fan-out)
    t = time.perf_counter()
    g_cpu_all = dist.new_group(list(range(world)), backend="gloo")
    w = dist.barrier(group=g_cpu_all, async_op=True)
    try:
        _wait(w, 60.0, "gloo barrier across all ranks")
        log("gloo_check_ok", s=round(time.perf_counter() - t, 3), gloo_ifname=os.environ.get("GLOO_SOCKET_IFNAME"))
    except Deadline as e:
        log("gloo_check_failed", msg=str(e), gloo_ifname=os.environ.get("GLOO_SOCKET_IFNAME"),
            hint="set GLOO_SOCKET_IFNAME to the interface that carries the inter-node address")
        dump("gloo_check_failed"); os._exit(4)
    g4 = dist.new_group(list(range(world)), backend=a.backend)            # the SP group, split from the default
    g3_ref = dist.new_group(list(range(world - 1)), backend=a.backend)    # reference only, made while all are alive
    log("init", backend=a.backend, cfg=cfg, host=res["host"])

    def run(x, group, n, r, steps, deadline):
        for _ in range(steps):
            for L in layers:
                x = layer_sp(L, x, group, n, r, deadline)
        return x

    # references
    x = run(x0.clone(), g4, world, rank, a.steps, 120.0); sync(dev)
    x_ref4 = x
    if rank < world - 1:
        t = time.perf_counter()
        x_ref3 = run(x0.clone(), g3_ref, world - 1, rank, a.steps, 120.0); sync(dev)
        res["sp3_step_s"] = (time.perf_counter() - t) / a.steps
    if rank == 0:
        x = x0.clone()
        for _ in range(a.steps):
            for L in layers:
                x = layer_sp1(L, x)
        sync(dev); x_ref1 = x
        res["control_sp4_vs_sp1"] = cmp(x_ref4, x_ref1); res["control_sp3_vs_sp1"] = cmp(x_ref3, x_ref1)
        log("references", sp4_vs_sp1=res["control_sp4_vs_sp1"], sp3_vs_sp1=res["control_sp3_vs_sp1"])
    dist.barrier()

    # failure run
    x = x0.clone()
    for s in range(a.steps):
        if rank == fail_rank and s == a.fail_step:
            log("fail.sigstop_self", step=s); dump("stopped")
            os.kill(os.getpid(), signal.SIGSTOP)
            time.sleep(1); return 0
        x_step = x.clone()
        try:
            t_step = time.perf_counter()
            for L in layers:
                x = layer_sp(L, x, g4, world, rank, a.deadline_s)
        except Deadline as e:
            t_miss = time.perf_counter()
            res["T_detect_s"] = t_miss - t_step
            log("deadline_miss", step=s, msg=str(e))
            break
    else:
        log("no_failure_seen"); dump("no_failure"); os._exit(1)

    # survivors: targeted abort of the stalled group
    t = time.perf_counter()
    if a.backend == "nccl":
        c10d._abort_process_group(g4)
    res["T_abort_s"] = time.perf_counter() - t
    log("abort_done", T_abort_s=round(res["T_abort_s"], 3))
    # membership through the default store; rank 0 (the store host) is the leader
    t = time.perf_counter()
    store.set(f"ft/e1/alive/{rank}", "1")
    if rank == 0:
        time.sleep(a.settle_s)
        members = [r for r in range(world) if store.check([f"ft/e1/alive/{r}"])]
        store.set("ft/e1/members", ",".join(map(str, members)))
    store.wait(["ft/e1/members"])
    members = [int(v) for v in store.get("ft/e1/members").decode().split(",")]
    res["T_rendezvous_s"] = time.perf_counter() - t; res["members"] = members
    log("membership", members=members, T_rendezvous_s=round(res["T_rendezvous_s"], 3))
    t = time.perf_counter()
    g3, cpu3, unbound = survivor_group(members, a.backend, dev)
    res["T_group_s"] = time.perf_counter() - t; res["device_unbound_for_group"] = unbound
    n, r = len(members), members.index(rank)
    # the new CPU group carries Python objects from the leader, as SGLang's scheduler does
    obj = [{"request": "next", "from": rank}] if rank == 0 else [None]
    dist.broadcast_object_list(obj, src=members[0], group=cpu3)
    res["cpu_group_broadcast_ok"] = obj[0] == {"request": "next", "from": 0}
    log("survivor_group", n=n, r=r, T_group_s=round(res["T_group_s"], 3), cpu_broadcast_ok=res["cpu_group_broadcast_ok"])
    # recompute step s and finish at SP=3 from the boundary state
    t = time.perf_counter()
    x = x_step
    for L in layers:
        x = layer_sp(L, x, g3, n, r, 120.0)
    sync(dev); res["T_first_sp3_step_s"] = time.perf_counter() - t
    x = run(x, g3, n, r, a.steps - s - 1, 120.0); sync(dev)
    res["steps_completed_sp3"] = a.steps - s
    res["cont_vs_sp3_ref"] = cmp(x, x_ref3); res["cont_vs_sp4_ref"] = cmp(x, x_ref4)
    if rank == 0:
        res["cont_vs_sp1_ref"] = cmp(x, x_ref1)
        ctrl = max(res["control_sp4_vs_sp1"]["rel_l2"], res["control_sp3_vs_sp1"]["rel_l2"])
        ok = res["cont_vs_sp3_ref"]["exact"] or res["cont_vs_sp3_ref"]["rel_l2"] <= max(ctrl, 1e-6)
        res["verdict"] = "PASS" if ok and res["cpu_group_broadcast_ok"] and res["steps_completed_sp3"] >= 2 else "FAIL"
        log("result", verdict=res["verdict"], cont_vs_sp3=res["cont_vs_sp3_ref"], cont_vs_sp4=res["cont_vs_sp4_ref"],
            cont_vs_sp1=res["cont_vs_sp1_ref"], T_detect=res["T_detect_s"], T_abort=res["T_abort_s"],
            T_rendezvous=res["T_rendezvous_s"], T_group=res["T_group_s"], T_first_sp3_step=res["T_first_sp3_step_s"],
            sp3_step=res.get("sp3_step_s"), device_unbound=unbound)
    # fence: a survivor on the failed rank's host kills it (after the work, so a supervisor does not tear us down early)
    host, pid = store.get(f"pid/{fail_rank}").decode().split(":")
    if host == socket.gethostname() and rank == min(m for m in members if store.get(f"pid/{m}").decode().split(":")[0] == host):
        try:
            os.kill(int(pid), signal.SIGCONT); os.kill(int(pid), signal.SIGKILL); log("fenced", pid=int(pid))
        except ProcessLookupError:
            pass
    dump("normal")
    os._exit(0)


if __name__ == "__main__":
    main()
