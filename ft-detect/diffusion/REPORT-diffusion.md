# Failure-detection latency kill test — Phase 2 (SGLang Diffusion, 2 GPUs, sequence parallel)

## 1. Stack

- SGLang `0.5.19` (sglang.multimodal_gen) at `/home/ckim151/venv-ft/lib/python3.12/site-packages/sglang`, commit `None`; torch `2.13.0+cu129`, torch NCCL PG default `600.0` s
- env at recon: `{"SGLANG_DIFFUSION_IPC_A2A": "false", "NCCL_P2P_DISABLE": "1", "CUDA_VISIBLE_DEVICES": "0,1"}`

## 2. Detectors and timeouts present in the installed diffusion runtime

| finding | value | file:line | note |
|---|---|---|---|
| dist_timeout_default | 3600 | sglang/multimodal_gen/runtime/server_args/server_args.py:307 | torch.distributed PG timeout; the only timeout on the NCCL fallback path |
| scheduler_rpc_timeout_default | None | sglang/multimodal_gen/runtime/server_args/server_args.py:308 | HTTP -> rank-0 scheduler ZMQ RCVTIMEO; None = wait forever |
| watchdog_timeout_arg |  | NOT FOUND | ABSENT in diffusion ServerArgs if NOT FOUND: no step watchdog exists |
| num_gpus_default | 1 | sglang/multimodal_gen/runtime/server_args/server_args.py:270 |  |
| health_handler | Report readiness for normal inference traffic. | sglang/multimodal_gen/runtime/entrypoints/http_server.py:161 | readiness flag only |
| health_body | if not request.app.state.server_warmup_done.is_set():        | sglang/multimodal_gen/runtime/entrypoints/http_server.py:162 | 200 forever once warm; never consults the engine |
| health_generate_body | Compatibility readiness endpoint; no generation is issued. | sglang/multimodal_gen/runtime/entrypoints/http_server.py:256 | no generation issued |
| rpc_rcvtimeo_set | socket.setsockopt(zmq.RCVTIMEO, timeout_ms) | sglang/multimodal_gen/runtime/scheduler_client.py:68 | only when a timeout is configured |
| rpc_ping_timeout_ms | 2000 | sglang/multimodal_gen/runtime/scheduler_client.py:210 | liveness ping used by the client, not by /health |
| max_consecutive_recv_errors | 3 | sglang/multimodal_gen/runtime/managers/scheduler.py:180 | rank-0 loop exits after this many recv errors |
| recv_error_msg | f"Error receiving requests in scheduler event loop | sglang/multimodal_gen/runtime/managers/scheduler.py:1205 |  |
| exec_error_msg | f"Error executing request in scheduler event loop | sglang/multimodal_gen/runtime/managers/scheduler.py:1237 |  |
| slave_result_recv | results.append(pipe.recv()) | sglang/multimodal_gen/runtime/managers/scheduler.py:1298 | rank 0 blocks here on a hung slave; no timeout |
| pdeathsig | kill_itself_when_parent_died() | sglang/multimodal_gen/runtime/managers/gpu_worker.py:1192 | workers get SIGKILL when the HTTP process dies |
| worker_shutdown_msg | f"Worker {rank}: Shutdown complete." | sglang/multimodal_gen/runtime/managers/gpu_worker.py:1237 |  |
| worker_dead_msg | f"Rank {rank_offset + i} scheduler is dead | sglang/multimodal_gen/runtime/launch_server.py:236 | only checked during startup |
| worker_proc_name | f"sglang-diffusionWorker-{rank}" | sglang/multimodal_gen/runtime/launch_server.py:192 |  |
| ipc_a2a_default | true | sglang/multimodal_gen/envs.py:290 | 2-rank Ulysses all-to-all over CUDA IPC |
| ipc_a2a_timeout_ms | 10000.0 | sglang/multimodal_gen/envs.py:292 | the ONLY sub-minute timeout on the per-step collective path |
| a2a_torch_fallback | dist.all_to_all_single( | sglang/multimodal_gen/runtime/distributed/device_communicators/base_device_communicator.py:189 | when IPC a2a is off/unavailable: torch PG op, covered by dist_timeout |
| denoise_step_def | def _run_denoising_step( | sglang/multimodal_gen/runtime/pipelines_core/stages/denoising.py:1494 | wrapped by launch_wrapped_diff.py for per-step timestamps |
| denoise_loop | for step_index, t_host in enumerate(timesteps_cpu) | sglang/multimodal_gen/runtime/pipelines_core/stages/denoising.py:1944 |  |

Reading (from the code; the runs below test it): there is no per-step watchdog and no rank-death poll after startup; `/health` is a readiness flag that stays 200 once warm; the HTTP->scheduler RPC waits forever by default; rank 0 blocks without timeout on a hung peer's result pipe; the only sub-minute timeout anywhere is the CUDA-IPC all-to-all used by 2-rank Ulysses (10 s), and the NCCL fallback path is covered only by `--dist-timeout` (3600 s).

## 3. T_detect per case and per detector (s after t_inject)

| run | case | target_rank | dist_timeout | t_first_symptom | t_stall_engine | t_detect_engine | engine_detector | t_detect_health | t_detect_gen1 | t_detect_gen1_sustained3 | t_server_dead | n_inflight_hung | n_new_errored | ttft_p50_before_ms | ttft_p99_before_ms | no_engine_detect | no_self_teardown | not_steady |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| diff_A_r1_1_20261002-203027 | A | 1 | default | -0.209 |  | 600.23 | log:nccl_timeout@rank0.log |  | 58.925 | 179.0 |  | 0 | 235 | 2491.7 | 8161.6 | False | True | False |
| diff_A_r1_1_20261002-210331 | A | 1 | default | -0.156 |  | 600.217 | log:nccl_timeout@rank0.log |  | 58.98 | 179.12 |  | 0 | 235 | 2596.1 | 8228.6 | False | True | False |
| diff_A_r1_2_20261002-213757 | A | 1 | default | -0.189 |  | 600.246 | log:nccl_timeout@rank0.log |  | 58.947 | 179.1 |  | 1 | 234 | 2509.4 | 8175.4 | False | True | False |
| diff_B_r1_1_20261002-221223 | B | 1 | default | -0.101 |  | 600.275 | log:nccl_timeout@rank0.log |  | 59.033 | 179.189 |  | 1 | 234 | 2636.1 | 8183.5 | False | True | False |
| diff_B_r1_2_20261002-225055 | B | 1 | default | -0.156 |  | 600.208 | log:nccl_timeout@rank0.log |  | 58.976 | 179.165 |  | 1 | 234 | 2591.0 | 8242.9 | False | True | False |
| diff_B_r1_3_20261002-232927 | B | 1 | default | -0.193 |  | 600.211 | log:nccl_timeout@rank0.log |  | 58.945 | 179.1 |  | 1 | 234 | 2494.9 | 8197.6 | False | True | False |

Median over runs (`no_detect_runs` = runs where nothing fired within MAX_DETECT_WAIT; `teardown` = HTTP server gone on its own):

| case | runs | engine_med | engine_min | engine_max | no_detect_runs | health_med | gen1_med | teardown_med | no_self_teardown_runs | hung_med | new_err_med |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 3 | 600.23 | 600.22 | 600.25 | 0 |  | 58.95 |  | 3 | 0.0 | 235.0 |
| B | 3 | 600.21 | 600.21 | 600.28 | 0 |  | 58.98 |  | 3 | 1.0 | 234.0 |

Note on `TTFT`: images are not streamed, so these are end-to-end latencies; `gen1` is a 1-step 256x256 request with a 60 s client timeout.

## 4. Timeout floors (no injection)

### mixed

| run | rate | rpc_timeout | dist_timeout | duration_s | n_requests | n_errors | error_kinds | ttft_p50_ms | ttft_p99_ms | spurious_kill | first_engine_event | health_non200_changes | gen1_max_consecutive_bad | launch_failed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| diff_floor_mixed_rpc_T60_20261003-000800 | 0.376 | 60 | default | 603.409 | 230 | 0 |  | 3409.4 | 15507.6 | False |  | 0 | 0 | False |
| diff_floor_mixed_rpc_T30_20261003-001900 | 0.376 | 30 | default | 603.452 | 230 | 0 |  | 3268.6 | 15559.9 | False |  | 0 | 0 | False |
| diff_floor_mixed_rpc_T10_20261003-003001 | 0.376 | 10 | default | 603.737 | 230 | 27 | http 500: 'Internal Server Error' | 2781.9 | 9658.0 | False |  | 0 | 2 | False |

Smallest clean `--scheduler-rpc-timeout`: **10** s; smallest clean `--dist-timeout`: **None** s (an RPC timeout 'false positive' here means a healthy request was cut off: it is a request-latency timeout, not a step timeout).

Hypothetical per-step watchdog floor from measured step durations:

| shape | pixels_frames | steps | requests | step_p50_ms | step_p99_ms | step_max_ms | gap_max_ms | outliers_gt5x_p50 | max_is_cold_shape | step_max_warm_ms | per_shape_floor_s | per_shape_floor_warm_s | slack_vs_global_x |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 256x256 | 65536 | 3048 | 1524 | 43.2 | 47.4 | 253.5 |  | 2 | True | 57.6 | 0.51 | 0.12 | 2.1 |
| 512x512 | 262144 | 8316 | 462 | 66.2 | 68.5 | 72.0 | 72.1 | 0 | False | 72.0 | 0.14 | 0.14 | 7.6 |
| 1024x1024 | 1048576 | 2916 | 162 | 222.5 | 225.1 | 230.2 | 230.5 | 0 | True | 230.0 | 0.46 | 0.46 | 2.3 |
| 1536x1536 | 2359296 | 1188 | 66 | 532.9 | 536.3 | 536.7 | 536.8 | 0 | False | 536.7 | 1.07 | 1.07 | 1.0 |

Global floor across shapes: **1.07 s**; smallest per-shape floor: **0.14 s** (ratio 8x).

### harsh (40% 1536x1536 @ 50 steps + 3x bursts)

_no data_

Smallest clean `--scheduler-rpc-timeout`: **None** s; smallest clean `--dist-timeout`: **None** s (an RPC timeout 'false positive' here means a healthy request was cut off: it is a request-latency timeout, not a step timeout).

Hypothetical per-step watchdog floor from measured step durations:

_no step_stats CSV; run step_stats.py over the floor runs_

## 5. Verdict

| case | runs | runs with no engine detection | T_detect engine (median s, detected runs) | T_detect any component (median s) | T_evict (median s) |
|---|---|---|---|---|---|
| A | 3 | 0 | 600.2 | 58.9 |  |
| B | 3 | 0 | 600.2 | 59.0 |  |

Criteria: T_detect >= 60 s AND floor >= 10 s. Hypothetical step-watchdog global floor from step times: **1.07 s**. If the engine never detected within the wait (no_detect runs), T_detect is bounded below by MAX_DETECT_WAIT and the first condition holds trivially; the verdict then rests on whether any timeout that *could* be configured has a floor above 10 s. Write the GO/KILL line in results/notes.md after reading sections 3-4; this generator does not guess it.

## 6. Workarounds and observations

- Verdict (Phase 2, SGLang Diffusion 0.5.19, Z-Image-Turbo, 2x L40S, --sp-degree 2 --ulysses-degree 2): **GO by the stated criteria.** T_detect at defaults = 600.2 s for both case A (kill -9) and case B (SIGSTOP), 3 runs each (range 600.21-600.28). Floor on the only configurable detection knob (--scheduler-rpc-timeout, request-level): 30 s clean, 10 s cut 27/230 healthy requests (12% false positives) -> floor 30 s >= 10 s.
- A == B: the surviving rank waits inside the SP all-to-all, so a dead peer and a stopped peer look identical; the only thing that fires is torch's ProcessGroupNCCL op timeout at its DEFAULT 600 s on the SP subgroup (PG ID 2, ALLTOALL_BASE). --dist-timeout 3600 applies to the default group only and did not change this. Nothing in the diffusion stack polls worker liveness after startup.
- No eviction: after the timeout the worker aborts (~60 s later), but the HTTP server keeps /health=200 and keeps queueing requests indefinitely (no_self_teardown in 6/6 runs; 234-235 requests lost per run in the 600 s window). The 1-step probe only "detected" via its own 60 s client timeout (59 s), because it queues behind the hung request.
- /health never left 200 in any run (readiness flag only); /liveness likewise.
- Per-step durations are stable and linear in pixels: 256^2 43 ms, 512^2 66 ms, 1024^2 222 ms, 1536^2 533 ms (p99 ~ max). A per-step watchdog, if one existed, would have a warm global floor of 1.07 s (2x safety) but a per-shape floor from 0.14 to 1.07 s (8x spread); one global value is 8x slack on small shapes. Cold-shape first steps were seen at 5.4 s (512^2, calibration run), which such a watchdog would also have to tolerate.
- IPC all-to-all path (SGLANG_DIFFUSION_IPC_A2A=true, default) is unusable on this host: the first request hung in the first step, the 10 s IPC timeout never fired, torch's heartbeat monitor killed the worker after 480 s, and the HTTP server served /health=200 for the next ~2 h with dead workers. All measurements use SGLANG_DIFFUSION_IPC_A2A=false (NCCL via SHM, NCCL_P2P_DISABLE=1); the IPC-path timeout is therefore unmeasured here.
- Deviations from the cookbook: the Z-Image-Turbo NVIDIA recipe is single-GPU; this experiment uses the documented --sp-degree/--ulysses-degree flags at 2 to force a per-step collective between ranks. Steps fixed at the model's 9 (LOAD_ARGS="--steps 9,9,9"); rate 0.376 req/s (70% of 0.537 saturation); harsh profile not run (1536^2 already in mixed).
