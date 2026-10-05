"""B2 feasibility: is the GPU still usable from a *separate* process while a rank on it is stuck?

Runs the same bf16 matmul / memset / D2H benchmark as the in-process coexistence probe, but in
its own process and CUDA context. Compare against a run with no hang.

    python gpu_bench.py --device 0 --out bench.json
"""

from __future__ import annotations

import argparse
import json
import time

import torch


def bench(n_iter=30, size=4096):
    dev = torch.device("cuda")
    out = {"device_name": torch.cuda.get_device_name(dev)}
    tw = time.perf_counter
    t = tw(); a = torch.randn(size, size, device=dev, dtype=torch.bfloat16); b = torch.randn(size, size, device=dev, dtype=torch.bfloat16)
    c = torch.empty(size, size, device=dev, dtype=torch.bfloat16); torch.cuda.synchronize(); out["alloc_init_ms"] = (tw() - t) * 1e3
    torch.matmul(a, b, out=c); torch.cuda.synchronize()
    e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
    t = tw(); e0.record()
    for _ in range(n_iter):
        torch.matmul(a, b, out=c)
    e1.record(); out["launch_ms"] = (tw() - t) * 1e3; e1.synchronize(); out["matmul_ms"] = e0.elapsed_time(e1) / n_iter
    d = torch.empty(256 * 1024 * 1024 // 2, device=dev, dtype=torch.bfloat16)
    e2 = torch.cuda.Event(enable_timing=True); e3 = torch.cuda.Event(enable_timing=True)
    e2.record(); d.zero_(); e3.record(); e3.synchronize(); out["memset_256MiB_ms"] = e2.elapsed_time(e3)
    h = torch.empty(64 * 1024 * 1024 // 2, dtype=torch.bfloat16, pin_memory=True)
    e4 = torch.cuda.Event(enable_timing=True); e5 = torch.cuda.Event(enable_timing=True)
    e4.record(); h.copy_(d[: h.numel()], non_blocking=True); e5.record(); e5.synchronize(); out["d2h_64MiB_ms"] = e4.elapsed_time(e5)
    free, total = torch.cuda.mem_get_info()
    out["free_MiB"] = free // 2**20; out["total_MiB"] = total // 2**20
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    torch.cuda.set_device(a.device)
    t0 = time.perf_counter()
    res = {"t_start": time.time(), "cuda_init_ms": None, "runs": []}
    torch.cuda.init(); torch.zeros(1, device="cuda"); torch.cuda.synchronize()
    res["cuda_init_ms"] = (time.perf_counter() - t0) * 1e3
    for _ in range(a.repeat):
        res["runs"].append(bench())
    print(json.dumps(res))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
