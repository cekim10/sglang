"""Test 1: steady-state cost of failure-containable issuing.

Two ranks run a Ulysses-shaped layer loop: per layer, compute -> all-to-all -> attention-sized
compute -> all-to-all -> compute. Two issuing modes:
  baseline  : synchronous all_to_all_single (the current stream waits on the NCCL stream); the CPU
              issues the next kernels immediately (run-ahead, as every torch SP implementation does)
  contained : async all_to_all_single; the CPU polls work.is_completed() and issues the next
              dependent kernel only after the collective has finished (no launch can ever queue
              behind an unfinished collective, so the process stays recoverable)
Per mode: wall time per step, GPU busy time (events around compute), the GPU gap between
collective completion and the next compute, and O = T_contained / T_baseline - 1.

    ./run_pair.sh discipline_bench.py --shape zimage   (tokens 4096, hidden 3840, heads 30)
    ./run_pair.sh discipline_bench.py --shape wan      (tokens 8192, hidden 3072, heads 24)
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F

SHAPES = {
    "zimage": {"tokens": 4096, "hidden": 3840, "heads": 30, "layers": 30},
    "wan": {"tokens": 8192, "hidden": 3072, "heads": 24, "layers": 30},
    "small": {"tokens": 1024, "hidden": 1024, "heads": 8, "layers": 8},
}


class Layer:
    """Deterministic weights; one Ulysses attention block: proj -> a2a -> SDPA (local heads) -> a2a -> out proj -> MLP."""

    def __init__(self, hidden, heads, dev, gen):
        self.heads, self.hidden, self.dd = heads, hidden, hidden // heads
        k = lambda *s: (torch.randn(*s, generator=gen, device=dev, dtype=torch.float32) * 0.02).to(torch.bfloat16)
        self.wqkv = k(hidden, 3 * hidden); self.wo = k(hidden, hidden); self.w1 = k(hidden, 4 * hidden); self.w2 = k(4 * hidden, hidden)


def layer_forward(layer: Layer, x_local, world, sync_op, poll_us, stats):
    """x_local: [S/world, hidden] tokens owned by this rank. Returns new x_local."""
    S_loc, H, hd = x_local.shape[0], layer.heads, layer.dd
    qkv = x_local @ layer.wqkv                                   # [S_loc, 3H*hd]
    qkv = qkv.view(S_loc, 3, H, hd)
    # all-to-all: sequence-sharded -> head-sharded. send [world, S_loc, 3, H/world, hd]
    send = qkv.view(S_loc, 3, world, H // world, hd).permute(2, 0, 1, 3, 4).contiguous()
    recv = torch.empty_like(send)
    t = _a2a(recv, send, sync_op, poll_us, stats)
    # recv: [world(seq chunk), S_loc, 3, H/world, hd] -> full sequence for local heads
    full = recv.permute(2, 0, 1, 3, 4).reshape(3, world * S_loc, H // world, hd)
    q, k, v = full[0], full[1], full[2]                           # [S, H/world, hd]
    o = F.scaled_dot_product_attention(q.transpose(0, 1).unsqueeze(0), k.transpose(0, 1).unsqueeze(0), v.transpose(0, 1).unsqueeze(0))
    o = o.squeeze(0).transpose(0, 1).contiguous()                 # [S, H/world, hd]
    send2 = o.view(world, S_loc, H // world, hd).contiguous()
    recv2 = torch.empty_like(send2)
    _a2a(recv2, send2, sync_op, poll_us, stats)
    attn = recv2.permute(1, 0, 2, 3).reshape(S_loc, H * hd)       # back to sequence-sharded, all heads
    x_local = x_local + attn @ layer.wo
    x_local = x_local + F.silu(x_local @ layer.w1) @ layer.w2
    return x_local


def _a2a(recv, send, sync_op, poll_us, stats):
    stream = torch.cuda.current_stream()
    e_before = torch.cuda.Event(enable_timing=True); e_before.record(stream)
    if sync_op:
        dist.all_to_all_single(recv, send)          # stream waits on NCCL; CPU returns at once
        work = None
    else:
        work = dist.all_to_all_single(recv, send, async_op=True)
        t0 = time.perf_counter()
        n = 0
        while not work.is_completed():               # containment: nothing is issued behind an unfinished collective
            n += 1
            if poll_us > 0:
                time.sleep(poll_us / 1e6)
        stats["poll_wait_s"] += time.perf_counter() - t0
        stats["polls"] += n
        work.wait()                                  # orders the current stream after the (now finished) collective
    e_after = torch.cuda.Event(enable_timing=True); e_after.record(stream)
    stats["a2a_events"].append((e_before, e_after))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--world", type=int, default=2)
    ap.add_argument("--shape", choices=sorted(SHAPES), default="wan")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--poll-us", type=float, default=0.0, help="sleep between polls in contained mode (0 = busy poll)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = SHAPES[a.shape]
    torch.cuda.set_device(0)
    dist.init_process_group("nccl", rank=a.rank, world_size=a.world)
    dev = torch.device("cuda", 0)
    gen = torch.Generator(device=dev).manual_seed(1234)
    layers = [Layer(cfg["hidden"], cfg["heads"], dev, gen) for _ in range(cfg["layers"])]
    S_loc = cfg["tokens"] // a.world
    x0 = (torch.randn(S_loc, cfg["hidden"], generator=gen, device=dev, dtype=torch.float32)).to(torch.bfloat16)
    results = {"shape": a.shape, "cfg": cfg, "rank": a.rank, "modes": {}}

    for mode in ("baseline", "contained", "baseline", "contained"):   # alternate to cancel drift; keep the last of each
        sync_op = mode == "baseline"
        # warm-up
        x = x0.clone()
        for L in layers[:2]:
            x = layer_forward(L, x, a.world, sync_op, a.poll_us, {"poll_wait_s": 0, "polls": 0, "a2a_events": []})
        torch.cuda.synchronize(); dist.barrier()
        step_times, gpu_busy, gaps, poll_wait = [], [], [], []
        for s in range(a.steps):
            stats = {"poll_wait_s": 0.0, "polls": 0, "a2a_events": []}
            x = x0.clone()
            torch.cuda.synchronize()
            e_start = torch.cuda.Event(enable_timing=True); e_end = torch.cuda.Event(enable_timing=True)
            t0 = time.perf_counter(); e_start.record()
            for L in layers:
                x = layer_forward(L, x, a.world, sync_op, a.poll_us, stats)
            e_end.record(); e_end.synchronize()
            wall = time.perf_counter() - t0
            total_gpu = e_start.elapsed_time(e_end) / 1e3
            # time the stream spent inside collectives (between the before/after events) = not computing
            in_a2a = sum(b.elapsed_time(e) for b, e in stats["a2a_events"]) / 1e3
            step_times.append(wall); gpu_busy.append(total_gpu - in_a2a); gaps.append(in_a2a); poll_wait.append(stats["poll_wait_s"])
            dist.barrier()
        results["modes"][mode] = {
            "step_wall_s_median": statistics.median(step_times), "step_wall_s_min": min(step_times),
            "gpu_compute_s_median": statistics.median(gpu_busy), "stream_in_collective_s_median": statistics.median(gaps),
            "cpu_poll_wait_s_median": statistics.median(poll_wait), "a2a_per_step": 2 * cfg["layers"],
        }
        if a.rank == 0:
            print(f"[{mode:9s}] step {statistics.median(step_times) * 1e3:8.1f} ms  (min {min(step_times) * 1e3:7.1f})  "
                  f"compute {statistics.median(gpu_busy) * 1e3:7.1f} ms  in-collective {statistics.median(gaps) * 1e3:7.1f} ms  "
                  f"cpu-poll {statistics.median(poll_wait) * 1e3:6.1f} ms", flush=True)
    b, c = results["modes"]["baseline"], results["modes"]["contained"]
    results["overhead_O"] = c["step_wall_s_median"] / b["step_wall_s_median"] - 1
    results["overhead_O_min"] = c["step_wall_s_min"] / b["step_wall_s_min"] - 1
    if a.rank == 0:
        O = results["overhead_O"]
        verdict = "PASS (<=3%)" if O <= 0.03 else "PASS/interesting (3-10%)" if O <= 0.10 else "GRAY (10-20%)" if O <= 0.20 else "naive discipline KILL (>20%)"
        print(f"\nshape={a.shape} tokens={cfg['tokens']} hidden={cfg['hidden']} layers={cfg['layers']} a2a/step={2 * cfg['layers']} poll_us={a.poll_us}")
        print(f"O = T_contained/T_baseline - 1 = {100 * O:+.1f}% (by min: {100 * results['overhead_O_min']:+.1f}%)  -> {verdict}")
        print(f"per collective: extra {(c['step_wall_s_median'] - b['step_wall_s_median']) / (2 * cfg['layers']) * 1e6:.0f} us; "
              f"stream-in-collective {c['stream_in_collective_s_median'] / (2 * cfg['layers']) * 1e6:.0f} us (contained) vs "
              f"{b['stream_in_collective_s_median'] / (2 * cfg['layers']) * 1e6:.0f} us (baseline)")
        results["verdict"] = verdict
    with open(a.out, "w") as f:
        json.dump(results, f, indent=1)
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
