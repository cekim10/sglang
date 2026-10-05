"""Test 2: in-process SP2 -> SP1 continuation after a peer failure (standalone, real tensors).

Model of a replicated-state Ulysses block (the Wan case from 3a): both ranks hold the full hidden
state x [S, hidden]; each layer projects its own token half, all-to-all to head sharding, SDPA over
the full sequence for its heads, all-to-all back, all-gather of the attention output so x stays
replicated, then residual + MLP. N steps of L layers from a fixed x0 with deterministic weights.

Protocol on rank 0 (contained issuing: nothing is launched behind an unfinished collective):
  1. reference_sp1 : all N steps on rank 0 alone (the SP1 math)              -> x_ref
  2. reference_sp2 : all N steps with both ranks                              -> x_sp2
  3. failure run   : both ranks restart from x0; rank 1 SIGSTOPs itself at the start of step k;
                     rank 0's first all-to-all of step k never completes; rank 0 polls with a
                     deadline, aborts the process group, then recomputes step k from the full x it
                     holds at SP1 and finishes steps k+1..N at SP1 in the same process/context -> x_cont
Measured: T_detect (deadline), T_abort, T_first_sp1_step (recompute of step k), T_remaining, and
x_cont vs x_ref / x_sp2 (exact, max-abs, rel-L2). PASS = several SP1 steps complete in-process
with x_cont within the SP1<->SP2 control of x_ref.

    MAX_S=200 ./run_pair.sh sp_switch_probe.py --shape wan --steps 12 --fail-step 5 --deadline-s 2
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
import torch.nn.functional as F

from discipline_bench import SHAPES, Layer

T0 = time.perf_counter()
LOG = []


def log(ev, **kw):
    rec = {"t": round(time.perf_counter() - T0, 4), "ev": ev, **kw}
    LOG.append(rec)
    print(f"[{rec['t']:8.3f}] {ev} {json.dumps(kw, default=str) if kw else ''}", flush=True)


class Deadline(Exception):
    pass


def a2a_contained(recv, send, deadline_s):
    work = dist.all_to_all_single(recv, send, async_op=True)
    t0 = time.perf_counter()
    while not work.is_completed():
        if time.perf_counter() - t0 > deadline_s:
            raise Deadline(f"all-to-all not complete after {deadline_s}s")
    work.wait()


def layer_sp2(layer: Layer, x_full, rank, world, deadline_s):
    S, H, hd = x_full.shape[0], layer.heads, layer.dd
    S_loc = S // world
    x_local = x_full[rank * S_loc:(rank + 1) * S_loc]
    qkv = (x_local @ layer.wqkv).view(S_loc, 3, H, hd)
    send = qkv.view(S_loc, 3, world, H // world, hd).permute(2, 0, 1, 3, 4).contiguous()
    recv = torch.empty_like(send)
    a2a_contained(recv, send, deadline_s)
    full = recv.permute(2, 0, 1, 3, 4).reshape(3, S, H // world, hd)
    q, k, v = full[0], full[1], full[2]
    o = F.scaled_dot_product_attention(q.transpose(0, 1).unsqueeze(0), k.transpose(0, 1).unsqueeze(0), v.transpose(0, 1).unsqueeze(0))
    o = o.squeeze(0).transpose(0, 1).contiguous()                 # [S, H/world, hd]
    send2 = o.view(world, S_loc, H // world, hd).contiguous()
    recv2 = torch.empty_like(send2)
    a2a_contained(recv2, send2, deadline_s)
    attn_local = recv2.permute(1, 0, 2, 3).reshape(S_loc, H * hd)  # this rank's tokens, all heads
    gathered = [torch.empty_like(attn_local) for _ in range(world)]
    work = dist.all_gather(gathered, attn_local, async_op=True)
    t0 = time.perf_counter()
    while not work.is_completed():
        if time.perf_counter() - t0 > deadline_s:
            raise Deadline("all-gather not complete")
    work.wait()
    attn_full = torch.cat(gathered, dim=0)
    x_full = x_full + attn_full @ layer.wo
    x_full = x_full + F.silu(x_full @ layer.w1) @ layer.w2
    return x_full


def layer_sp1(layer: Layer, x_full):
    S, H, hd = x_full.shape[0], layer.heads, layer.dd
    qkv = (x_full @ layer.wqkv).view(S, 3, H, hd)
    q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
    o = F.scaled_dot_product_attention(q.transpose(0, 1).unsqueeze(0), k.transpose(0, 1).unsqueeze(0), v.transpose(0, 1).unsqueeze(0))
    attn = o.squeeze(0).transpose(0, 1).reshape(S, H * hd)
    x_full = x_full + attn @ layer.wo
    x_full = x_full + F.silu(x_full @ layer.w1) @ layer.w2
    return x_full


def run_steps_sp1(layers, x, steps):
    for _ in range(steps):
        for L in layers:
            x = layer_sp1(L, x)
    return x


def cmp(a, b):
    d = (a.float() - b.float()).abs()
    return {"exact": bool(torch.equal(a, b)), "max_abs": float(d.max()), "rel_l2": float(d.norm() / (b.float().norm() + 1e-12))}


def watchdog(max_s, out):
    time.sleep(max_s)
    log("watchdog.exit")
    json.dump({"log": LOG, "exit": "watchdog"}, open(out, "w"), indent=1)
    os._exit(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--world", type=int, default=2)
    ap.add_argument("--shape", choices=sorted(SHAPES), default="wan")
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--fail-step", type=int, default=5)
    ap.add_argument("--deadline-s", type=float, default=2.0)
    ap.add_argument("--max-s", type=float, default=240.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    threading.Thread(target=watchdog, args=(a.max_s, a.out), daemon=True).start()
    cfg = dict(SHAPES[a.shape]); cfg["layers"] = a.layers
    torch.cuda.set_device(0)
    dist.init_process_group("nccl", rank=a.rank, world_size=a.world)
    dev = torch.device("cuda", 0)
    gen = torch.Generator(device=dev).manual_seed(1234)
    layers = [Layer(cfg["hidden"], cfg["heads"], dev, gen) for _ in range(cfg["layers"])]
    x0 = torch.randn(cfg["tokens"], cfg["hidden"], generator=gen, device=dev, dtype=torch.float32).to(torch.bfloat16)
    log("init", rank=a.rank, cfg=cfg, steps=a.steps, fail_step=a.fail_step)
    res = {"cfg": cfg, "steps": a.steps, "fail_step": a.fail_step, "rank": a.rank}

    # 1) SP1 reference (rank 0 only; rank 1 idles) and SP1 step time
    if a.rank == 0:
        torch.cuda.synchronize(); t = time.perf_counter()
        x_ref = run_steps_sp1(layers, x0.clone(), a.steps)
        torch.cuda.synchronize(); res["sp1_step_s"] = (time.perf_counter() - t) / a.steps
        log("reference_sp1_done", step_s=round(res["sp1_step_s"], 4))
    dist.barrier()

    # 2) SP2 reference with both ranks
    x = x0.clone()
    torch.cuda.synchronize(); t = time.perf_counter()
    for s in range(a.steps):
        for L in layers:
            x = layer_sp2(L, x, a.rank, a.world, 60.0)
    torch.cuda.synchronize(); res["sp2_step_s"] = (time.perf_counter() - t) / a.steps
    x_sp2 = x
    log("reference_sp2_done", step_s=round(res["sp2_step_s"], 4))
    if a.rank == 0:
        res["control_sp2_vs_sp1"] = cmp(x_sp2, x_ref)
        log("control", **res["control_sp2_vs_sp1"])
    dist.barrier()

    # 3) failure run
    x = x0.clone()
    for s in range(a.steps):
        if s == a.fail_step and a.rank == 1:
            log("rank1.sigstop_self", step=s)
            json.dump({"log": LOG, "exit": "stopped"}, open(a.out, "w"), indent=1)
            os.kill(os.getpid(), signal.SIGSTOP)
            log("rank1.resumed"); return 0
        if a.rank == 1 and s < a.fail_step:
            for L in layers:
                x = layer_sp2(L, x, a.rank, a.world, 60.0)
            dist.barrier()
            continue
        # rank 0
        x_step_start = x.clone()                      # the protected boundary state (replicated full x)
        try:
            t_step0 = time.perf_counter()
            for L in layers:
                x = layer_sp2(L, x, a.rank, a.world, a.deadline_s)
            if s < a.fail_step:
                dist.barrier()
        except Deadline as e:
            t_detect = time.perf_counter()
            log("deadline_miss", step=s, after_s=round(t_detect - t_step0, 3), msg=str(e))
            from torch.distributed import distributed_c10d as c10d
            t = time.perf_counter()
            try:
                c10d._abort_process_group()
                res["T_abort_s"] = time.perf_counter() - t
                log("abort_done", T_abort_s=round(res["T_abort_s"], 3))
            except Exception as ex:
                res["T_abort_s"] = None; log("abort_failed", error=repr(ex)[:200])
            res["T_detect_s"] = t_detect - t_step0
            # in-process continuation at SP1 from the boundary state, same tensors/context
            t = time.perf_counter()
            x = x_step_start
            for L in layers:
                x = layer_sp1(L, x)
            torch.cuda.synchronize()
            res["T_first_sp1_step_s"] = time.perf_counter() - t
            log("first_sp1_step_done", T_first_sp1_step_s=round(res["T_first_sp1_step_s"], 3))
            t = time.perf_counter()
            x = run_steps_sp1(layers, x, a.steps - s - 1)
            torch.cuda.synchronize()
            res["T_remaining_s"] = time.perf_counter() - t
            res["steps_completed_sp1"] = a.steps - s
            log("continuation_done", steps_sp1=res["steps_completed_sp1"], T_remaining_s=round(res["T_remaining_s"], 3))
            break
    if a.rank == 0:
        res["cont_vs_sp1_ref"] = cmp(x, x_ref)
        res["cont_vs_sp2_ref"] = cmp(x, x_sp2)
        ctrl = res["control_sp2_vs_sp1"]["rel_l2"]
        ok = res.get("steps_completed_sp1", 0) >= 2 and res["cont_vs_sp1_ref"]["rel_l2"] <= max(2 * ctrl, 1e-3)
        res["verdict"] = "PASS" if ok else "FAIL"
        log("result", verdict=res["verdict"], cont_vs_sp1=res["cont_vs_sp1_ref"], cont_vs_sp2=res["cont_vs_sp2_ref"],
            control=res["control_sp2_vs_sp1"], T_detect=res.get("T_detect_s"), T_abort=res.get("T_abort_s"),
            T_first_sp1_step=res.get("T_first_sp1_step_s"), sp1_step=res.get("sp1_step_s"), sp2_step=res.get("sp2_step_s"))
    json.dump({"log": LOG, **res, "exit": "normal"}, open(a.out, "w"), indent=1, default=str)
    os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
