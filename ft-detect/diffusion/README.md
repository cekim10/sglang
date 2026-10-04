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
export LOAD_ARGS="--steps 9,9,9"                   # turbo models (Z-Image-Turbo): keep the model's step count; omit for FLUX.1-dev

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

## Workload-variability characterization (the kill-test input, no fault injection)

Question: is there one static step timeout that is both fast and false-positive-free across
image and video workloads? `characterize.sh` runs a shape matrix on one model, N sequential
requests per shape (first = cold sample), and records every denoising step and every pipeline
stage with the request's shape. `step_stats.py` turns that into per-workload deadlines
(`D_w = 2 x p99.9(warm step)`), cold-start maxima by cause, and the non-denoising stage table;
`merge_floor.py` merges models into one `D_global / D_w` table and a log-axis chart.

```bash
# image (Z-Image-Turbo, 9 steps)
SHAPES="256x256@9,512x512@9,1024x1024@9,1536x1536@9" PER_SHAPE=5 IDLE_PROBE=120 ./characterize.sh zimage

# video (Wan; frame counts must be 4k+1; 480p and 720p; short / medium / long)
export MODEL=Wan-AI/Wan2.2-TI2V-5B-Diffusers            # fits 2x48 GB without offload; A14B needs --dit-cpu-offload
SHAPES="832x480x17@30,832x480x49@30,832x480x81@30,1280x720x17@30,1280x720x49@30,1280x720x81@30" PER_SHAPE=3 ./characterize.sh wan22

python merge_floor.py results/char_*.csv --md results/floor_merged.md
```

Video requests go through `POST /v1/videos` (multipart form, returns a queued job) and are
polled on `GET /v1/videos/{id}` until `completed`/`failed`; `--shapes` with a frame count
switches the generator to that path automatically. Add `--fps` via `LOAD_ARGS` if the model's
default fps/seconds mapping matters.

## Phase 3a: resume-state inventory (what a checkpoint would carry, and who already has it)

With `FT_INVENTORY=1`, every rank records once per request, right after denoising step
`FT_INVENTORY_STEP` (default 2) completes: the latent tensor (shape, bytes, checksum), the
scheduler class and any multi-step solver history, conditioning tensors, generator state, the
total resume bundle `S_state`, and the measured cost of copying that bundle to pinned host memory
(`T_pin`) and serializing it (`T_save`). `inventory.py` compares ranks: equal latent checksums
mean the surviving SP rank already holds the full `x_t`; `did_sp_shard_latents` or differing
checksums mean recovery needs the dead rank's shard. From the code (0.5.19): image models shard
latents spatially across SP ranks (Z-Image: along H or W); video models shard along latent time
only when it divides the SP degree, so 17/49/81 frames (latent 5/13/21) stay replicated while
e.g. 85 frames (latent 22) shard.

```bash
unset LOAD_ARGS
FT_INVENTORY=1 SHAPES="512x512@9,1536x1536@9" PER_SHAPE=2 ./characterize.sh inv_zimage          # MODEL=Tongyi-MAI/Z-Image-Turbo
FT_INVENTORY=1 SHAPES="512x512@9,832x480x17@9,832x480x81@9,832x480x85@9,1280x704x81@9" PER_SHAPE=2 ./characterize.sh inv_wan22_5b
cat results/inventory_inv_zimage.md results/inventory_inv_wan22_5b.md
```

## Phase 3c: containment cost (abort a stuck collective, fail the request, keep the rank)

`FT_ABORT_PROBE=1` arms a measurement thread in every rank: if a request is in flight and no
quantum (stage or step) has completed for `FT_ABORT_DEADLINE_S` s, it calls
`torch.distributed.distributed_c10d._abort_process_group` (`FT_ABORT_MODE=all|sp|world`) and
timestamps: deadline miss, abort returned, the exception surfacing in the blocked main thread,
the request ending, the rank back at its receive loop, CUDA memory before/after. Nothing marks the
replica unroutable; that cost is measured separately and is tiny (an asyncio event in the HTTP
process). Run case B with the probe and read the timeline:

```bash
unset LOAD_ARGS; export MODEL=Wan-AI/Wan2.2-TI2V-5B-Diffusers
FT_ABORT_PROBE=1 FT_ABORT_DEADLINE_S=10 MAX_DETECT_WAIT=120 POST_GRACE=60 ./run_case_diff.sh B 1 2
TORCH_NCCL_ASYNC_ERROR_HANDLING=0 FT_ABORT_PROBE=1 FT_ABORT_DEADLINE_S=10 MAX_DETECT_WAIT=120 POST_GRACE=60 ./run_case_diff.sh B 1 2
cat results/containment_B.md
```

Expected outcomes to distinguish: (a) abort returns in ms, the main thread raises, the request
fails, the rank lives and later requests fail fast -> containment is bounded by the deadline;
(b) abort returns but the main thread stays blocked (kernel spinning) -> containment needs a
different primitive; (c) the rank dies (torch async error handling tears it down) -> containment
equals restart. The SIGSTOPped peer is SIGCONTed by stop.py at the end of the run.

Variants after the first result (in-process `_abort_process_group` never returned; the rank was
terminated ~60 s later by torch's own abort/dump timeout):

```bash
# NCCL non-blocking communicators: does abort return now?
TORCH_NCCL_USE_COMM_NONBLOCKING=1 FT_ABORT_PROBE=1 FT_ABORT_DEADLINE_S=10 MIN_WARMUP=45 STEADY_WAIT=60 MAX_DETECT_WAIT=120 POST_GRACE=60 ./run_case_diff.sh B 1 2
# process-level containment floor: the rank exits at the deadline instead of aborting
FT_ABORT_MODE=exit FT_ABORT_PROBE=1 FT_ABORT_DEADLINE_S=10 MIN_WARMUP=45 STEADY_WAIT=60 MAX_DETECT_WAIT=120 POST_GRACE=60 ./run_case_diff.sh B 1 2
```

## Phase 3b-1: trajectory portability (SP=2 boundary state -> fresh SP=1 process)

`portability.sh <tag> <shape> <steps> <k>` runs one fixed request (`one_request.py`, fixed
prompt and seed) under five fresh servers: SP=2 saving the state after step k and the final
latents; SP=1 reference twice (noise floor); SP=1 resumed from the saved state at step k+1 with
the full multi-step solver history; and the same with the history reset (low-order restart).
The wrapper does the save/skip/inject (`FT_TRAJ_*`); SGLang is unmodified.
`trajectory_compare.py` reports exact equality, max-abs / relative-L2 distance of the final
latents, output hash equality, restore time, first resumed step time and remaining completion time.

```bash
unset LOAD_ARGS
MODEL=Wan-AI/Wan2.2-TI2V-5B-Diffusers ./portability.sh wan_480p81 832x480x81 9 3     # UniPC: history matters
MODEL=Tongyi-MAI/Z-Image-Turbo        ./portability.sh zimage_1024 1024x1024 9 3     # Euler: control without history
cat results/traj_wan_480p81/compare.md results/traj_zimage_1024/compare.md
```

Gate (agreed in advance): `sp1_full` matches `sp1_ref` within the SP2<->SP1 control (`sp2_save` vs
`sp1_ref`) and the run-to-run floor (`sp1_ref2`) -> GO. Bit-exactness lost only to collective
ordering is not a KILL. `sp1_lower` quality is a secondary result.

## Phase 3b-2 prerequisite: restart decomposition from existing logs

`startup_decomp.py runs/...` splits launch -> /health 200 into spawn, import, dist init,
load (weights + H2D + device setup, not separable in the logs) and warmup, using our own
records and the runtime's log lines. New runs also write `ready.json`.

```bash
python startup_decomp.py runs/diff_char_* runs/traj_* --md results/startup_decomp.md
```
