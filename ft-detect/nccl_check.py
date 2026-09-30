"""60-second NCCL sanity check for the GPUs in CUDA_VISIBLE_DEVICES, outside SGLang.

Spawns one process per visible GPU, runs init_process_group(nccl) with a short
timeout, then one all_reduce and one broadcast. Prints where it hangs if it does.

    python nccl_check.py                       # defaults: timeout 60 s, NCCL_DEBUG=WARN
    NCCL_DEBUG=INFO python nccl_check.py       # verbose: shows chosen interface, P2P/SHM transport
    NCCL_P2P_DISABLE=1 python nccl_check.py    # try with PCIe P2P off
"""

from __future__ import annotations

import datetime
import os
import socket
import sys
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank: int, world: int, port: int, timeout_s: int):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    t0 = time.time()

    def say(msg):
        print(f"[rank{rank} +{time.time() - t0:5.1f}s] {msg}", flush=True)

    say(f"gpu={torch.cuda.get_device_name(rank)} init_process_group(nccl, timeout={timeout_s}s) ...")
    dist.init_process_group("nccl", rank=rank, world_size=world, timeout=datetime.timedelta(seconds=timeout_s))
    say("process group ready; all_reduce ...")
    x = torch.ones(1024, device="cuda") * (rank + 1)
    dist.all_reduce(x)
    torch.cuda.synchronize()
    say(f"all_reduce ok (sum={x[0].item():.0f}, expected {world * (world + 1) / 2:.0f}); broadcast ...")
    y = torch.full((1024,), float(rank), device="cuda")
    dist.broadcast(y, src=0)
    torch.cuda.synchronize()
    say(f"broadcast ok (got {y[0].item():.0f}); 64 MB all_reduce x10 ...")
    big = torch.ones(16 * 1024 * 1024, device="cuda")
    t1 = time.time()
    for _ in range(10):
        dist.all_reduce(big)
    torch.cuda.synchronize()
    say(f"done, {(time.time() - t1) / 10 * 1e3:.1f} ms per 64 MB all_reduce")
    dist.destroy_process_group()
    sys.stdout.flush()
    os._exit(0)  # do not wait on NCCL/CUDA teardown; the measurement is done


def main():
    n = torch.cuda.device_count()
    if n < 2:
        print(f"need >= 2 visible GPUs, have {n} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})")
        return 2
    timeout_s = int(os.environ.get("NCCL_CHECK_TIMEOUT", "60"))
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    print(f"torch {torch.__version__} nccl {'.'.join(map(str, torch.cuda.nccl.version()))} gpus={n} "
          f"NCCL_DEBUG={os.environ['NCCL_DEBUG']} NCCL_P2P_DISABLE={os.environ.get('NCCL_P2P_DISABLE')} "
          f"NCCL_SOCKET_IFNAME={os.environ.get('NCCL_SOCKET_IFNAME')} NCCL_IB_DISABLE={os.environ.get('NCCL_IB_DISABLE')}")
    for i in range(n):
        for j in range(n):
            if i != j:
                print(f"  can_device_access_peer({i},{j}) = {torch.cuda.can_device_access_peer(i, j)}")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    t0 = time.time()
    ctx = mp.spawn(worker, args=(n, port, timeout_s), nprocs=n, join=False)
    ok = ctx.join(timeout=timeout_s * 3 + 30)
    if not ok:
        print(f"HANG: workers did not finish in {timeout_s * 3 + 30}s; killing", flush=True)
        for p in ctx.processes:
            p.kill()
        return 1
    print(f"OK in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
