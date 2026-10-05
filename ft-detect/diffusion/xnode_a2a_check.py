"""Can two 2-GPU machines run Ulysses SP=4 at a usable speed? Times all_to_all_single across nodes.

Run with torchrun on both machines (2 processes each). Sizes: the all-to-all the Wan 480p x 81 run
issued in Test 3 (37.7M bf16 elements, from the NCCL watchdog line) and two smaller ones. Prints
the transport NCCL picked (with NCCL_DEBUG=INFO), per-call time and bandwidth, and an estimate of
how much slower a denoising step gets than on one machine.

    # elves-01 (node 0)                                    # elves-02 (node 1)
    torchrun --nnodes 2 --nproc-per-node 2 --node-rank 0 --master-addr elves-01 --master-port 29555 xnode_a2a_check.py
    torchrun --nnodes 2 --nproc-per-node 2 --node-rank 1 --master-addr elves-01 --master-port 29555 xnode_a2a_check.py
    # single-machine reference (2 ranks, elves-01 only)
    torchrun --nproc-per-node 2 xnode_a2a_check.py
"""

from __future__ import annotations

import os
import socket
import time

import torch
import torch.distributed as dist

SIZES = {"wan480p81": 37_739_520, "quarter": 37_739_520 // 4, "small": 1 << 20}
LOCAL_STEP_S = 0.72          # Wan 480p x 81 step at SP=2 on one elves node (Test 3)
A2A_PER_STEP = 120           # rough: 2 per block x 30 blocks x 2 (CFG); order of magnitude only


def main():
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local))
    rank, world = dist.get_rank(), dist.get_world_size()
    host = socket.gethostname()
    hosts = [None] * world
    dist.all_gather_object(hosts, host)
    if rank == 0:
        print(f"world {world}: {hosts}; NCCL {'.'.join(map(str, torch.cuda.nccl.version()))}", flush=True)
    res = {}
    for name, n in SIZES.items():
        n -= n % world
        x = torch.randn(n, device="cuda", dtype=torch.bfloat16)
        y = torch.empty_like(x)
        for _ in range(3):
            dist.all_to_all_single(y, x)
        torch.cuda.synchronize(); dist.barrier()
        reps = 10 if n > (1 << 22) else 50
        t = time.perf_counter()
        for _ in range(reps):
            dist.all_to_all_single(y, x)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t) / reps
        res[name] = dt
        if rank == 0:
            moved = n * 2 * (world - 1) / world        # bytes each rank sends to the others
            print(f"{name:10s} {n * 2 / 2**20:8.1f} MiB/rank  {dt * 1e3:9.2f} ms/call  {moved / dt / 1e9:6.2f} GB/s per rank", flush=True)
    if rank == 0:
        extra = A2A_PER_STEP * res["wan480p81"]
        print(f"rough Wan 480p x 81 step at SP={world}: compute ~{LOCAL_STEP_S * 2 / world:.2f} s + all-to-all ~{extra:.1f} s "
              f"(order of magnitude; compare with the single-machine run of this script)", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
