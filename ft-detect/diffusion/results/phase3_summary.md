# Failure handling in distributed diffusion serving: what the measurements settled

Host: elves-01 (2x NVIDIA L40S 48 GB on different NUMA nodes), SGLang 0.5.19 (`sglang.multimodal_gen`),
torch 2.13.0+cu129, NCCL 2.29.7. Models: Tongyi-MAI/Z-Image-Turbo (6B image, Euler) and
Wan-AI/Wan2.2-TI2V-5B (video, UniPC), sequence parallel over 2 GPUs (`--sp-degree 2 --ulysses-degree 2`).
Host quirks: `NCCL_P2P_DISABLE=1`, `SGLANG_DIFFUSION_IPC_A2A=false` (P2P across the NUMA boundary is broken).
LLM numbers (Phase 1, Qwen3-14B TP=2) are the contrast, not the subject. Raw tables: `REPORT.md` (LLM),
`REPORT-diffusion.md`, `floor_merged.md`, `inventory_*.md`, `containment_B.md`, `traj_*/compare.md`,
`startup_decomp.md`, `notes.md`.

## 1. Detection (Phase 2)

| rank failure | LLM stack | diffusion stack |
|---|---|---|
| kill -9 (A) | noticed 0.02-0.12 s; replica gone 65 s | noticed 600.2 s; replica never gone |
| SIGSTOP (B) | noticed 352 s (watchdog); gone 422 s | noticed 600.2 s; never gone |
| `/health` | 503 after 20 s of silence | readiness flag: 200 forever |
| requests lost per window | 338 | 235 |

The only thing that fires in the diffusion stack is torch's ProcessGroupNCCL op timeout at its default
600 s on the SP subgroup; `--dist-timeout` does not reach that group; there is no step or rank watchdog,
no RPC timeout, and nothing acts on the result.

## 2. Why a static timeout cannot be both fast and safe (Phase 2b)

Per-step time is a near-deterministic function of shape (p99.9 within 1-3% of p50) but spans
43 ms (256^2 image) to 712 ms (480p x 81 video); the largest execution quantum of a video request
is the VAE decode: 6.0 s at 480p x 81, 12.8 s at 704p x 81. One static deadline safe for every served
workload is 25.6 s (31 s with cold starts) while the smallest workload needs 0.54 s: 48x slack,
>25 s absolute. A per-stage static table still carries 10x (decode) / 16x (step) shape dependence.
Cold cost exists only for the first request after launch (3.0-3.6 s step, 7 s decode), not per shape.

## 3. Recovery state is cheap and portable (Phase 3a, 3b-1)

| | Z-Image 1536^2 | Wan 5B 480p x 81 | Wan 5B 704p x 81 |
|---|---|---|---|
| latents on serving rank | 0.56 MiB, sharded across SP ranks | 6.0 MiB, replicated | 13.5 MiB, replicated |
| multi-step solver history | none (Euler) | 18 MiB (UniPC) | 40.6 MiB |
| per-request static conditioning | 0.37 MiB | 16 MiB | 16 MiB |
| device -> pinned host, per step | 0.14 ms (0.03% of step) | 10.5 ms (1.5%) | 22 ms (1.1%) |

Wan pipelines keep the loop-level latent replicated on every SP rank (DiT-internal sharding);
image pipelines shard it, but the peer-owned part is <= 0.6 MiB. Resuming an SP=2 request in a fresh
SP=1 process from the state after step 3 reproduces the SP=1 reference within the SP2<->SP1 control:
Z-Image rel-L2 2.04e-2 (control 2.04e-2), Wan 8.99e-2 with the full UniPC history (control 9.02e-2),
1.03e-1 with a low-order restart. Restore costs 2-18 ms from a 0.5-25 MiB file.

## 4. Containment: the process is lost, the GPU is not (Phase 3c, B3-1, B2)

With the peer SIGSTOPped, the surviving rank's main thread is not inside the collective at all: the
all-to-all sits on the GPU waiting, the thread keeps issuing the next layers, and it finally blocks in
an ordinary cuBLAS launch when the context's launch queue is full. From then on:

| primitive | result |
|---|---|
| `torch.distributed._abort_process_group` (ncclCommAbort) from a side thread | never returns; the rank dies only via torch's own 600 s + 60 s path |
| `TORCH_NCCL_USE_COMM_NONBLOCKING=1` | normal steps fail (`NCCL Error 7: operation in progress`); unusable |
| side thread, side stream, one in-place memset on a pre-allocated buffer | launch blocks: the whole CUDA context can no longer launch |
| exiting the process at the deadline | gone in 0.1-8 s (context teardown); later requests fail in 0 s; `/health` still 200 |
| a *separate* process on the same GPU during the hang | works: context init 14 ms (idle 11 ms), matmul 1.07 ms vs 0.51 ms idle (time-slicing with the spinning NCCL kernel), 26.5 GB free of 44.5 GB |

Restart cost of a fresh rank (`startup_decomp.md`): spawn + torch/sglang import ~9 s, weight load + H2D +
device setup 15-17 s, warmup 0.5-2 s (Z-Image) to ~12 s (Wan 5B); 24-43 s total.

## 5. What follows for the design

The surviving process cannot compute, abort, or even copy its state out once a collective is stuck;
the GPU underneath it can run another process at roughly half speed until the stuck one is killed; the
trajectory state is tens of MiB and resumes exactly in a different process and parallel degree if it
was written to host memory *before* the failure, which costs about 1% of a step.

Mechanism that these facts admit, with every component motivated by a measurement:

1. **Progress deadlines per execution quantum** (stage, and denoising step inside the denoising stage),
   from a per-(model, shape, frames, stage) table of warm quantum times with a cold flag for the first
   request after launch; the runtime already knows every input. (Phase 2b: static timeouts are 48x slack;
   intra-workload variance 1-3%.)
2. **Boundary protection**: after every quantum, copy the dynamic state (latent + solver history) to
   pinned host memory; conditioning once at admission; for sharded configurations exchange the peer's
   shard (<= 0.6 MiB). (3a: <= 1% of step.)
3. **Containment on deadline miss**: mark the replica unroutable, stop admission, and terminate the
   stuck rank processes; do not try to abort the communicator. (3c, B3-1.)
4. **Continuation in a fresh process on the same GPU**, at the same or a lower SP degree, from the last
   protected boundary. (3b-1, B2.) The open design variable is the time to a ready process:
   24-43 s cold today; a warm standby per GPU is possible when memory allows (5B: 26 GB free during the
   hang), and the import/load/warmup split says where any faster path would have to act.

Claims to keep narrow: replication of loop-level state is a property of the Wan pipelines, not of
diffusion in general; the SP2<->SP1 numerical control is large for Wan (9% rel-L2) and is reported as
the floor for any "exactness" statement; all numbers are from one host with a broken P2P path, which
forced the NCCL-over-SHM transport and left the 10 s IPC all-to-all timeout unmeasured.
