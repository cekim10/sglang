# ft-detect / diffusion: Phase 2 on SGLang Diffusion (sglang.multimodal_gen)

Same question as Phase 1, different serving stack: how long until anything notices a dead
(case A, `kill -9`) or hung (case B, `SIGSTOP`) GPU rank of a sequence-parallel diffusion
replica, and what the zero-false-positive timeout floor is. Measurement only; no detector
is implemented.

Reuses `../common.py`, `../probe.py` (with `FT_API=diffusion`), `../inject.py`, `../stop.py`,
`../steady.py`, `../analyze.py`. New here:

| file | role |
|---|---|
| `recon_diff.py` | detectors/timeouts present in the installed diffusion runtime, with file:line; `--tree` |
| `launch_wrapped_diff.py` | stock `launch_server` with per-rank logs, PIDs, and one record per denoising step (shape + duration) |
| `launch_diff.sh` | start a replica (`--num-gpus $NGPU`, `$PARALLEL_ARGS`) and wait for readiness |
| `load_diff.py` | open-loop `/v1/images/generations` load; `mixed`, `harsh` (bursts), `video`; `--closed-loop` to calibrate |
| `calibrate_diff.sh` | saturation -> 70% rate, plus per-shape step times |
| `run_case_diff.sh` | injection driver (long `MAX_DETECT_WAIT` by default: see below) |
| `sweep_floor_diff.sh` | sweeps `--scheduler-rpc-timeout` / `--dist-timeout`; collects step times |
| `step_stats.py` | per-shape step-duration distribution -> hypothetical step-watchdog floor, global vs per shape |
| `report_diff.py` | `REPORT-diffusion.md` |

## What the code read says to expect (verify with `recon_diff.py`)

- No per-step watchdog exists (`watchdog_timeout` is not a diffusion ServerArg).
- `/health` is a readiness flag: 503 until warmup, then 200 forever. `/health_generate` issues nothing.
- `--scheduler-rpc-timeout` defaults to None: the HTTP server waits forever on rank 0.
- Rank 0 collects slave results with a plain `pipe.recv()`: no timeout.
- 2-rank Ulysses all-to-all goes over CUDA IPC with `SGLANG_DIFFUSION_IPC_A2A_TIMEOUT_MS=10000`; when IPC is
  off or unusable it falls back to `dist.all_to_all_single`, covered only by `--dist-timeout` (3600 s).
- Workers carry PDEATHSIG: if the HTTP process dies, ranks are SIGKILLed; the reverse has no monitor after startup.

So case B may be caught at ~10 s by the IPC all-to-all timeout, or at 3600 s by torch, or never, depending on
which path the collective takes. Measure both: `SGLANG_DIFFUSION_IPC_A2A=true` (default) and `=false`.

## Order of operations (elves-01: 2x L40S across NUMA nodes)

```bash
cd ~/sglang && git pull && cd ft-detect/diffusion
pip install "sglang[diffusion]==0.5.19"            # once; same venv as Phase 1

export CUDA_VISIBLE_DEVICES=0,1 NGPU=2 FT_PORT=30000 NCCL_P2P_DISABLE=1
export MODEL=black-forest-labs/FLUX.1-dev         # needs HF_TOKEN; ~33 GB. Alternatives below.
export PARALLEL_ARGS="--sp-degree 2 --ulysses-degree 2"   # per-step all-to-all between the two ranks

python recon_diff.py                                # Step 0 on the installed version
./calibrate_diff.sh mixed                           # prints "70% -> --rate X" and per-shape step times
export RATE=X

./run_case_diff.sh A 1 1                            # first-run check: rank logs, steps_rank0.jsonl 'ds' records, detection
./run_case_diff.sh A 1 5
MAX_DETECT_WAIT=4000 ./run_case_diff.sh B 1 1       # learn which timeout (if any) fires: 10 s IPC, 3600 s torch, or none
./run_case_diff.sh B 1 4                            # then the rest (lower MAX_DETECT_WAIT if run 1 was deterministic)
SGLANG_DIFFUSION_IPC_A2A=false ./run_case_diff.sh B 1 2   # NCCL fallback path

./sweep_floor_diff.sh mixed                         # SWEEP=rpc by default; also SWEEP=dist
./calibrate_diff.sh harsh && RATE=<harsh 70%> ./sweep_floor_diff.sh harsh

python report_diff.py                               # -> REPORT-diffusion.md; write the verdict line in results/notes.md
```

Model options for two 48 GB GPUs (check `$HF_HOME` first; anything over 20 GB is your call to download):
FLUX.1-dev (12B DiT + T5, gated, ~33 GB, 28-50 steps, good step-time spread across 512-1536 px);
FLUX.1-schnell (same size, Apache, 4 steps: too few steps per request for step-level measurement);
Tongyi-MAI/Z-Image-Turbo (6B, Apache, ~8 steps, smallest download, in the cookbook with `--num-gpus 2`).

## Definitions that differ from Phase 1

- No streaming: `first_token_ts == last_token_ts` = response time; `TTFT` columns are end-to-end latency.
- `t_stall_engine` uses a 60 s step gap (`--gap 60`) because a single 1536x1536 step can take seconds.
- `n_inflight_hung` is small by construction (no batching: one running request plus the queue).
- The floor has two meanings: the swept RPC/dist timeouts (request-level), and the hypothetical step-watchdog
  floor from `step_stats.py` (what Phase 1 measured directly for the LLM). The second is the Phase 2 result.
