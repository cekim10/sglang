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

## B3-1: kernel coexistence probe (does the stuck rank's GPU still compute?)

`FT_ABORT_MODE=coexist` leaves the communicator alone. The probe thread measures a bf16 matmul
benchmark on a fresh CUDA stream (event-synchronised only) ~90 s after start-up as a baseline and
again at the progress deadline while the main thread is blocked in the SP all-to-all. Equal
timings mean the GPU, weights and trajectory remain usable for an in-process SP=1 continuation
(B3-2); a large slowdown or a hang means the stuck NCCL kernel starves the device.

```bash
FT_ABORT_PROBE=1 FT_ABORT_MODE=coexist FT_ABORT_DEADLINE_S=10 MIN_WARMUP=100 STEADY_WAIT=60 MAX_DETECT_WAIT=60 POST_GRACE=30 ./run_case_diff.sh B 1 1
python containment.py --events runs/diff_B_r1_1_$(date +%Y%m%d)-* | grep -E "baseline|during hang|deadline_miss"
```

## B2 feasibility: the GPU from another process during the hang

B3-1 showed the stuck rank's whole CUDA context stops launching kernels (the issuing thread
blocks in an ordinary cuBLAS launch behind the stuck all-to-all; a side stream cannot launch
even an in-place memset). `gpu_bench.py` answers the next question: can a *separate* process use
that GPU at normal speed while the poisoned one sits there?

```bash
CUDA_VISIBLE_DEVICES=0 python gpu_bench.py --out results/gpu_bench_idle.json          # no hang, reference
FT_EXTERNAL_BENCH=1 FT_ABORT_PROBE=1 FT_ABORT_MODE=coexist FT_ABORT_DEADLINE_S=10 MIN_WARMUP=100 STEADY_WAIT=60 MAX_DETECT_WAIT=60 POST_GRACE=60 ./run_case_diff.sh B 1 1
cat runs/diff_B_r1_1_*/external_bench_during_hang.json | python -c "import json,sys; d=json.load(sys.stdin); print(d['cuda_init_ms'], [r['matmul_ms'] for r in d['runs']], d['runs'][0]['free_MiB'])"
```

## Option-3 probe: issuing discipline vs the poisoned context

`run_launch_discipline.sh` runs two bare NCCL ranks (no SGLang). Rank 1 freezes after warm-up so
rank 0's next all-to-all can never finish. In `flood` mode rank 0 keeps launching on the same
stream (what a DiT forward does) and we count how many launches fit before `cudaLaunchKernel`
blocks; in `disciplined` mode rank 0 enqueues the collective, records an event, polls it without
issuing anything else, and after the deadline tries `_abort_process_group`, a fresh launch on the
main stream, and `destroy_process_group`. A side thread benchmarks its own stream throughout.
If the side stream stays healthy and the escape hatches return in `disciplined` mode, the
B3-1 result was a consequence of run-ahead issuing, not of the stuck collective itself.

```bash
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1 ./run_launch_discipline.sh both
```

## Test 1 / Test 2: failure-containable issuing and in-process SP2 -> SP1 continuation (standalone)

Both are bare `torch.distributed` programs run by `run_pair.sh` (rank 0 on the first visible GPU,
rank 1 on the second), using a Ulysses-shaped layer with deterministic weights; the SP2 math was
checked against the SP1 math on CPU (bit-exact).

- `discipline_bench.py`: steady-state cost of never launching behind an unfinished collective.
  `baseline` issues run-ahead after a synchronous all-to-all; `contained` polls `work.is_completed()`
  before the next dependent kernel. Reports `O = T_contained / T_baseline - 1` with the agreed gate
  (<=3% pass, 3-10% pass/interesting, 10-20% gray, >20% naive discipline fails).
- `sp_switch_probe.py`: rank 1 freezes at the start of step k; rank 0 hits its poll deadline, aborts
  the process group, recomputes step k from the replicated boundary state at SP1 in the same
  process and finishes the remaining steps; reports T_detect, T_abort, T_first_sp1_step and the
  continued result against SP1 and SP2 references.

```bash
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1 MAX_S=300 ./run_pair.sh discipline_bench.py --shape wan
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1 MAX_S=300 ./run_pair.sh discipline_bench.py --shape zimage
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1 MAX_S=240 OUT_DIR=results/pair_sp_switch_wan ./run_pair.sh sp_switch_probe.py --shape wan --steps 12 --fail-step 5 --deadline-s 2
```

## Test 3: failure-contained execution inside SGLang Diffusion (Wan, SP=2 -> SP=1 in-process)

Everything from Test 1 and Test 2, applied to the real server through `launch_wrapped_diff.py`
(monkeypatches only; SGLang is not modified):

- `FT_CONTAIN=1` wraps `torch.distributed.{all_to_all_single, all_gather_into_tensor, all_gather,
  all_reduce, broadcast}` (and the `all_gather_single` alias the diffusion runtime bound at import):
  every collective is issued with `async_op=True` and polled under `FT_CONTAIN_DEADLINE_S`
  (default 5 s inside a request, 900 s outside one), so the CPU never issues a dependent kernel
  behind an unfinished collective. A miss raises `PeerFailure` inside the DiT forward.
- The `_run_denoising_step` hook catches it and, in the same process: `_abort_process_group()`
  (all groups), shrinks every `GroupCoordinator` in `parallel_state` to world size 1 (their ops
  short-circuit at 1; the global `.rank` is kept so the executor's main-rank checks still hold),
  resets the DiT's cached `sp_size`, sets `enable_sequence_shard=False` on the request, no-ops
  `dist.barrier` and `broadcast_pyobj` (gloo cannot be aborted; the executor syncs after every stage
  and the scheduler fans requests out through them), kills the peer, and re-runs the same step.
  Wan's loop-level latents are replicated, so the boundary state is already complete locally.
- `FT_FAIL_STEP=k FT_FAIL_RANK=1 FT_FAIL_REQ=2`: rank 1 SIGSTOPs itself at step k of its second
  non-warmup request (the deterministic injection; inject.py case B is the time-based alternative).

```bash
export CUDA_VISIBLE_DEVICES=0,1 NCCL_P2P_DISABLE=1 SGLANG_DIFFUSION_IPC_A2A=false MODEL=Wan-AI/Wan2.2-TI2V-5B-Diffusers
MODE=sp1ref ./run_contain.sh                       # SP=1 reference: output + latency (NGPU=1)
MODE=ours   N=3 ./run_contain.sh                   # ref request -> failing request (contained) -> next request at SP=1
MODE=stock  ./run_contain.sh                       # optional: same injection without containment (bounded 700 s)
python contain_report.py runs/contain_ours_* --sp1ref 'runs/contain_sp1ref_*' --md results/test3_contain.md --csv results/test3_contain.csv
```

Each run records `T_detect` (freeze -> deadline miss), `T_abort`, `T_switch`, `T_recompute`, the
SP=1 step time inside the failed request, `T_added` (failing request minus the SP=2 reference in
the same server), the latency of the next request served by the surviving process, and the final
latents of every request. Output check: `relerr_vs_sp1ref` of the contained request against the
control `control_sp1_vs_sp2` (the SP=1 and SP=2 references differ numerically on their own).
PASS = the failing request completes with an error within the control and the next request is
served at SP=1 by the same process. If the rank dies right after the abort, retry with
`TORCH_NCCL_ASYNC_ERROR_HANDLING=0` (torch's documented setting for `_abort_process_group`).

### Test 3 run 1 finding and the abort matrix

First valid run (2026-10-05): the deadline fired at exactly 5.0 s after rank 1 froze, but
`_abort_process_group()` (all groups) never returned in the server; torch's NCCL watchdog then
killed rank 0 at 600 s and the job stayed unfinished. The option-3 probe, where the same call
returned in 0.6 s, used one lazily created group. SGLang initialises the default group eagerly
(`device_id`), so its subgroups are NCCL splits of it, and it adds gloo groups.
`abort_matrix_probe.py` rebuilds that layout and tries one variant per run, each bounded:

```bash
CUDA_VISIBLE_DEVICES=0,1 NCCL_P2P_DISABLE=1 ./run_abort_matrix.sh     # ~8 cases, <2 min each worst case
cat results/abort_matrix/summary.md
```

The wrapper now fences the peer first, aborts only the stuck group by default
(`ABORT=stuck|world|none`), bounds the abort (`ABORT_TIMEOUT_S`, stacks dumped to the rank log on
a hang), prints a timestamp per failover sub-step, and the failing request is bounded by
`REQ_TIMEOUT` (300 s) so a hang ends the run. Pick `ABORT` from the matrix.

### Test 3 run 2 finding: weight-sharded components break the shrink

With the stuck-group abort and fence-first the failover completed in 0.59 s and the failing request
finished correctly. The next request, served by the same process at SP=1, returned an output whose
final latents differ from the SP=1 reference by 1.15 (control 0.09). Cause: with tp=1 and sp>1,
`--encoder-parallel auto` folds the text encoder's weights across the SP group for the model's
lifetime, so after the shrink rank 0 encodes with half the weights and the per-layer all-reduce
short-circuits. The failing request was unaffected because its text conditioning was computed
before the failure. Containment therefore needs every component the survivor runs to be
weight-replicated (Ulysses SP for the DiT, `--encoder-parallel replicate|dp` for encoders);
`run_contain.sh` now launches with `ENCODER_PARALLEL=replicate` and the failover logs any
weight-sharded component it finds.

### Test 3 run 3 finding: activation-parallel VAE decode cached its degree

With the encoder replicated, the final latents of every request were identical: the SP=2 reference,
the contained request, the next request and the SP=1 reference (the old 0.09 control came from the
folded encoder). But the contained and next requests returned a different video file (~30% smaller).
The Wan VAE builds spatial-parallel decoder modules when the decode group has two ranks and caches
`world_size`/`rank` in them, so after the shrink it decoded only rank 0's half of the latent height
and the gather short-circuited. The failover now resets every module's cached `world_size`/`rank`/
`sp_size`, turns on the convs' built-in "spatial parallel decode disabled" mode, and clears the
caches keyed on group identity or shape. The report now checks the decoded videos too (frames x
height x width and mean abs pixel difference against the SP=1 reference, needs imageio).
Two classes of state therefore exist: weight-sharded components cannot shrink in place (replicate
them), while activation-parallel components can, if every cached degree is found and reset.

### Test 3 result (2026-10-05): PASS, 3 of 3 runs

Wan2.2-TI2V-5B, 832x480x81, 9 steps, SP=2 on 2x L40S (SHM transport); rank 1 SIGSTOPs at step 4;
`--encoder-parallel replicate`, fence first, abort of the stuck group only, 5 s deadline.

| | per run (3 runs) |
|---|---|
| freeze to deadline miss | 5.00 s |
| abort of the stuck group | 0.09-0.44 s |
| miss to SP=1 ready (fence, abort, shrink, cached-state reset) | 1.2-1.6 s |
| failed step re-run at SP=1 | 0.50-0.52 s |
| failing request latency (normal SP=2 request: 15.1 s) | 32.8 s |
| next request, same process at SP=1 | 23.7-24.2 s |
| final latents vs SP=1 and SP=2 references | identical (rel. error 0) |
| decoded video vs SP=1 reference | same 81x480x832; mean abs pixel diff 1.273 (SP=2 reference: 1.274) |

Measured alternatives on this host: stock SGLang loses the request (600 s to the torch watchdog,
replica never evicted); a restart at SP=1 needs 43 s to ready and 46.9 s for its first request
(sp1ref run), on top of detection, and the in-flight request is lost.
Conditions and limits: one failure mode (SIGSTOP between steps), one model and shape, two GPUs
without NVLink; after the failover the process stays at SP=1 (no re-expansion); aborting every
group hangs in this layout (not yet attributed, see the abort matrix); the 1.2-1.6 s switch is
dominated by three heap scans that could be done once at startup.

## Cluster oracle: restart vs trajectory migration vs in-place degradation (computation only)

`cluster_oracle.py` simulates R replicas x SP=N serving the measured Wan video mix at 50 steps, one
GPU failure per event, the same 5 s detection for every policy, and compares, against the same
arrivals without a failure: P1 retry + restart, P2 migrate the trajectory + restart (the strongest
baseline; survivors restart at SP=N-1 after a hard failure), P3 in-place SP=N -> SP=N-1, P3i the same
with re-expansion only when idle. Restart time is swept (40 s measured for 5B, up to 15 min), with
hard and transient failures, rho 0.3-0.9, and a fleet view over per-GPU failure rates.

```bash
python cluster_oracle.py --seeds 40                                  # results/cluster_oracle.md
python cluster_oracle.py --seeds 40 --mix long-video --md results/cluster_oracle_longvideo.md --csv results/cluster_oracle_longvideo.csv
python cluster_oracle.py --seeds 40 --scaling perfect --md results/cluster_oracle_perfect_scaling.md --csv results/cluster_oracle_perfect_scaling.csv
```

Result (2026-10-05): GRAY in all three variants under the gate fixed before the run. In-place
degradation helps only for hard failures at high load with slow restarts; at the measured 40 s
restart it is within 10-15% of migration + restart. At hardware failure rates the failure-attributable
SLO violations are below 1e-4 of requests for every detected policy; the 600 s non-detection of the
stock stack causes several times more damage than any detected policy.

## Native NCCL fault tolerance vs issuance discipline (collision test)

Does PyTorch/NCCL's own recovery (`_abort_process_group`, or `dist.shrink_group(..., SHRINK_ABORT)`
over `ncclCommShrink`, NCCL >= 2.27) recover the surviving rank even after the runtime has issued
dependent kernels behind the stuck collective? `native_ft_probe.py` reproduces SGLang's layout (eager
init, split subgroup), freezes rank 1, issues none / 1 / 8 / unbounded dependent kernels, and recovers
from a detector thread; each step is bounded.

```bash
CUDA_VISIBLE_DEVICES=0,1 NCCL_P2P_DISABLE=1 ./run_native_ft.sh      # 16 cases; results/native_ft/summary.md
```

Gate: if the survivor is recoverable with a native primitive after k1/k8/flood, the issuance
discipline is unnecessary (its core claim is killed). If only `none` recovers, the runtime's issuance
decides whether communication-layer fault tolerance stays usable.

### Result (2026-10-05): issuance core KILLED

torch 2.13.0+cu129, NCCL 2.29.7, blocking communicators (the nonblocking half of the matrix produced
no rows). `_abort_process_group(subgroup)` (ncclCommAbort) called from a detector thread recovered
the survivor in every case, 0.59-0.69 s, including after 1, 8 and unbounded dependent kernels: the
blocked main thread was released, new work on the same stream completed and
`torch.cuda.synchronize()` returned. `shrink_group(..., SHRINK_ABORT)` never returned within 20 s,
although the stuck kernel was gone afterwards.
This corrects the option-3 conclusion. The flood mode there never called abort from another thread,
so "the first launch behind a stuck collective makes the survivor unrecoverable" was not shown; the
main thread was blocked, not the context poisoned. The in-server hang of Test 3 run 1 came from
aborting every group in the eager-split layout; aborting only the stuck group is what made Test 3
work, not the issuance discipline.

## SP=4 -> SP=3: re-forming the surviving group (standalone probe)

`sp_shrink_probe.py` + `run_shrink.sh`: a replicated-state Ulysses block at SP=4 on two 2-GPU
machines; the last rank SIGSTOPs at step k; the survivors detect the stalled all-to-all, abort only
that group, agree on membership through the default store (rank 0 leads), build new NCCL and gloo
groups among themselves, and recompute step k and finish at SP=3. SGLang's eager device binding
makes torch split NCCL subgroups from the default communicator, which would need the dead rank, so
the survivor groups are built with the device unbound (members-only rendezvous). Local CPU check
(4 gloo ranks): PASS, membership [0, 1, 2], new groups in 25 ms, object broadcast over the new CPU
group works, continuation bit-exact against the SP=3, SP=4 and SP=1 references.
Two machines: elves-01 and elves-03 pass only 1500-byte frames between them although both NICs are
set to MTU 9200, so NCCL needs `LD_PRELOAD=netfix/mss_clamp.so` (unprivileged TCP MSS clamp) with
sockets (`NCCL_IB_DISABLE=1`); cross-node all-to-all is ~24x slower than within a machine.

### Test 3 on two nodes: SP=4 -> SP=3 inside SGLang Diffusion

When more than two SP ranks exist, the failover shrinks instead of falling back to one rank
(`FT_CONTAIN_SHRINK=1`, default): targeted abort of the stalled group; survivors register in the
default store and global rank 0 publishes the membership (`FT_SHRINK_SETTLE_S`, default 1 s); each
node fences its own dead ranks; survivor NCCL and gloo groups are built with the device unbound;
every coordinator that contained a dead rank is pointed at them with a new CudaCommunicator (the
SP coordinator's Ulysses fields too; pure Ulysses only), the VAE decode group becomes a singleton
with spatial-parallel decode off, the GPU worker's cached SP CPU group and torch's default-group
barrier are rebound, the DiT's cached `sp_size` becomes N-1, and the failed step is recomputed.
A failure of global rank 0 (store host and request ingress) is out of scope and raises.
`launch_diff.sh` takes `NNODES`/`NODE_RANK`/`DIST_INIT_ADDR`; `run_contain_worker.sh` runs node 1
and relaunches its workers each time node 0's store goes away.
