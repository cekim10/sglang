# Failure-detection latency kill test — Phase 1 (SGLang, TP=4)

## 1. Stack

- SGLang version: `0.5.19` at `/home/ckim151/venv-ft/lib/python3.12/site-packages/sglang`; git commit: `None`
- torch `2.13.0+cu129`, NCCL `2.29.7`, python `3.12.3`
- torch default process-group timeout: `1800.0` s; NCCL default: `600.0` s
- relevant env at recon time: `{"CUDA_VISIBLE_DEVICES": "0,1"}`

## 2. Timeout defaults found in the installed code

| finding | value | file:line | note |
|---|---|---|---|
| watchdog_timeout_default | 300 | sglang/srt/server_args.py:1196 |  |
| soft_watchdog_timeout_default | None | sglang/srt/server_args.py:1201 |  |
| dist_timeout_default | None | sglang/srt/server_args.py:921 | None -> torch default |
| watchdog_poll | self.watchdog_timeout / 2 | sglang/srt/utils/watchdog.py:147 | fires between 1.0x and 1.5x timeout |
| watchdog_fire_msg | f"{self.debug_name} watchdog timeout " | sglang/srt/utils/watchdog.py:154 |  |
| watchdog_post_fire_sleep | 5 | sglang/srt/utils/watchdog.py:162 | sleep before SIGQUIT to parent |
| subprocess_watchdog_interval | 1.0 | sglang/srt/utils/watchdog.py:181 | rank-death poll period |
| subprocess_watchdog_msg | f"Subprocess {name} (pid={proc.pid}) crashed " | sglang/srt/utils/watchdog.py:217 | logged by the HTTP-server process when a rank exits non-zero |
| health_check_timeout | 20 | sglang/srt/entrypoints/http_server.py:193 | /health and /health_generate 503 after this many s without any engine output |
| health_handler | @app.get("/health") | sglang/srt/entrypoints/http_server.py:662 |  |
| health_fail_msg | f"Health check failed. Server couldn't get a response from detokenizer for last " | sglang/srt/entrypoints/http_server.py:734 |  |
| gloo_cpu_group_timeout | 120 * 60 | sglang/srt/distributed/parallel_state.py:294 | CPU group used for the per-step TP request broadcast |
| device_group_timeout_passthrough | _MODEL_PARALLEL_GROUP_TIMEOUT = timeout | sglang/srt/distributed/parallel_state.py:2280 | None -> torch default NCCL pg timeout |
| scheduler_exception_msg | f"Scheduler hit an exception: {traceback}" | sglang/srt/managers/scheduler.py:5604 |  |
| run_batch_def | def run_batch( | sglang/srt/managers/scheduler.py:4013 |  |
| process_batch_result_def | def process_batch_result( | sglang/srt/managers/scheduler.py:4357 |  |
| get_next_batch_to_run_def | def get_next_batch_to_run( | sglang/srt/managers/scheduler.py:3335 |  |
| forward_ct_increment | self.forward_ct += 1 | sglang/srt/managers/scheduler.py:4019 |  |
| crash_pyspy_dump_default | True | sglang/srt/environ.py:382 | SIGQUIT handler runs py-spy on schedulers before killing the tree |
| crash_cuda_coredump_default | True | sglang/srt/environ.py:383 | SIGQUIT handler waits for CUDA coredumps before killing the tree |
| crash_cuda_coredump_wait_s | 60.0 | sglang/srt/environ.py:384 | seconds the handler sleeps even when no coredump is enabled |
| crash_handler_settle_sleep | 5 | sglang/srt/managers/tokenizer_manager.py:3109 | unconditional sleep at the top of the crash-dump path |
| sigquit_handler_msg | f"SIGQUIT received. {signum=}, {frame=}. It usually means one child failed." | sglang/srt/managers/tokenizer_manager.py:3637 |  |

Interpretation (from the Step 0 code read):
- The scheduler watchdog counts `forward_ct`, polls every `timeout/2`, so a hang is noticed 1.0x-1.5x the timeout after the last step, then sleeps 5 s and SIGQUITs the parent, which kills the whole tree. Detection == replica death; there is no 'unhealthy' state.
- A dead rank (non-zero exit code) is noticed by `SubprocessWatchdog` polling every 1 s in the HTTP-server process; a SIGSTOPped rank still counts as alive.
- `/health` and `/health_generate` share one handler: 200 as soon as *any* engine output arrives, 503 after `SGLANG_HEALTH_CHECK_TIMEOUT` (20 s) of silence.
- `--dist-timeout` defaults to None (torch default). TP all-reduce under load goes through custom all-reduce / pynccl, which torch's NCCL watchdog does not cover; the gloo CPU group for the per-step request broadcast has a hard-coded 2 h timeout.

## 3. T_detect per case and per detector (s after t_inject)

| run | case | target_rank | watchdog_timeout | t_first_symptom | t_stall_engine | t_detect_engine | engine_detector | t_detect_health | t_detect_gen1 | t_detect_gen1_sustained3 | t_server_dead | n_inflight_hung | n_new_errored | ttft_p50_before_ms | ttft_p99_before_ms | ttft_p50_during_ms | ttft_p99_during_ms | not_steady |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A_r1_1_20260929-231600 | A | 1 | default | 0.011 | -0.011 | 0.103 | log:scheduler_exception@rank0.log | 59.081 | 5.004 | 15.173 | 65.35 | 3 | 0 | 447.3 | 5064.8 |  |  | False |
| A_r1_2_20260929-235034 | A | 1 | default | 0.014 | -0.008 | 0.11 | log:scheduler_exception@rank0.log | 59.085 | 5.009 | 15.152 | 65.336 | 3 | 0 | 421.6 | 5057.0 |  |  | False |
| A_r1_3_20260930-002508 | A | 1 | default | 0.003 | -0.02 | 0.024 | log:scheduler_exception@rank0.log | 5.132 | 5.026 | 5.177 | 5.175 | 3 | 0 | 442.4 | 5041.0 |  |  | False |
| A_r1_4_20260930-005942 | A | 1 | default | 0.016 | -0.006 | 0.042 | log:scheduler_exception@rank0.log | 59.124 | 5.044 | 15.192 | 65.286 | 3 | 0 | 421.7 | 5054.0 |  |  | False |
| A_r1_5_20260930-013416 | A | 1 | default | 0.018 | -0.005 | 0.115 | log:scheduler_exception@rank0.log | 59.093 | 5.015 | 15.175 | 65.296 | 3 | 0 | 433.4 | 5041.5 |  |  | False |
| B_r1_1_20260930-095903 | B | 1 | default | 0.008 | -0.015 | 352.063 | log:watchdog_fire@rank0.log | 20.099 | 4.986 | 15.142 | 422.344 | 3 | 338 | 421.6 | 5071.3 |  |  | False |
| B_r1_2_20260930-103336 | B | 1 | default | 0.016 | -0.007 | 351.658 | log:watchdog_fire@rank0.log | 20.136 | 5.025 | 15.182 | 422.021 | 3 | 362 | 431.3 | 5044.5 |  |  | False |
| B_r1_3_20260930-110810 | B | 1 | default | 0.005 | 0.002 | 352.097 | log:watchdog_fire@rank0.log | 20.147 | 4.936 | 15.117 | 422.443 | 3 | 338 | 444.9 | 5046.7 |  |  | False |
| B_r1_4_20260930-114244 | B | 1 | default | 0.017 | -0.005 | 351.835 | log:watchdog_fire@rank0.log | 20.131 | 5.017 | 15.169 | 422.115 | 3 | 337 | 441.5 | 5048.1 |  |  | False |
| B_r1_5_20260930-121718 | B | 1 | default | 0.011 | -0.011 | 351.43 | log:watchdog_fire@rank0.log | 20.145 | 5.039 | 15.193 | 421.704 | 3 | 336 | 424.6 | 5030.0 |  |  | False |
| B_r1_1_20261001-193533 | B | 1 | default | -1.895 | 0.024 | 395.308 | log:watchdog_fire@rank0.log | 20.587 | 3.442 | 13.621 | 465.595 | 10 | 373 | 225.6 | 4171.1 |  |  | False |
| B_r1_2_20261001-201239 | B | 1 | default | -0.851 | 0.212 | 305.51 | log:watchdog_fire@rank0.log | 21.175 | 3.831 | 14.005 | 375.714 | 10 | 294 | 525.3 | 5074.0 |  |  | False |
| B_r1_3_20261001-204942 | B | 1 | default | -3.303 | -0.125 | 381.861 | log:watchdog_fire@rank0.log | 20.807 | 2.03 | 12.212 | 452.157 | 27 | 357 | 312.1 | 5429.9 |  |  | False |

Median over runs:

| case | watchdog_timeout | runs | engine_med | engine_min | engine_max | health_med | gen1_med | hung_med | new_err_med |
|---|---|---|---|---|---|---|---|---|---|
| A | default | 5 | 0.1 | 0.02 | 0.12 | 59.08 | 5.01 | 3.0 | 0.0 |
| B | default | 8 | 351.95 | 305.51 | 395.31 | 20.15 | 4.96 | 3.0 | 338.0 |

## 4. Timeout floor (no injection)

### mixed

| run | rate | watchdog_timeout | dist_timeout | duration_s | n_requests | n_errors | ttft_p50_ms | ttft_p99_ms | spurious_kill | first_engine_event | first_pid_death | health_non200_changes | gen1_max_consecutive_bad | launch_failed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| floor_mixed_together_T300_20261001-150902 | 0.96 | 300 | 300 | 1212.208 | 1191 | 0 | 545.9 | 8585.2 | False |  |  | 0 | 3 | False |
| floor_mixed_together_T60_20261001-153022 | 0.96 | 60 | 60 | 1207.528 | 1186 | 0 | 509.3 | 8392.8 | False |  |  | 0 | 3 | False |
| floor_mixed_together_T30_20261001-155138 | 0.96 | 30 | 30 | 1207.491 | 1186 | 0 | 512.1 | 8405.5 | False |  |  | 0 | 3 | False |
| floor_mixed_together_T10_20261001-161253 | 0.96 | 10 | 10 | 1212.193 | 1191 | 0 | 545.5 | 8587.6 | False |  |  | 0 | 3 | False |
| floor_mixed_together_T5_20261001-163413 | 0.96 | 5 | 5 | 1207.494 | 1186 | 0 | 503.6 | 8426.4 | False |  |  | 0 | 3 | False |
| floor_mixed_together_T2_20261001-165528 | 0.96 | 2 | 2 | 1207.551 | 1186 | 0 | 504.8 | 8440.6 | False |  |  | 0 | 3 | False |

Smallest clean setting: **2** s

### harsh (30% 32k prefill + 3x bursts)

Rows at different `rate` values are separate experiments: a rate above the profile's own 70% point is an overload run (TTFT in the hundreds of seconds, request errors), kept because it shows the watchdog's behaviour under overload, not as a 'realistic load' measurement.

| run | rate | watchdog_timeout | dist_timeout | duration_s | n_requests | n_errors | ttft_p50_ms | ttft_p99_ms | spurious_kill | first_engine_event | first_pid_death | health_non200_changes | gen1_max_consecutive_bad | launch_failed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| floor_harsh_together_T10_20261001-182608 | 0.96 | 10 | 10 | 1316.974 | 1285 | 623 | 400834.4 | 662099.0 | False |  |  | 0 | 10 | False |
| floor_harsh_together_T10_20261002-144357 | 0.39 | 10 | 10 | 1214.931 | 520 | 0 | 6956.9 | 27566.2 | False |  |  | 0 | 10 | False |
| floor_harsh_together_T2_20261001-191225 | 0.96 | 2 | 2 | 1320.075 | 1285 | 624 | 403224.8 | 666396.5 | False |  |  | 0 | 10 | False |
| floor_harsh_together_T2_20261002-152643 | 0.39 | 2 | 2 | 1214.937 | 520 | 0 | 7007.8 | 27670.4 | False |  |  | 0 | 10 | False |
| floor_harsh_together_T300_20261001-171643 | 0.96 | 300 | 300 | 1316.707 | 1301 | 640 | 399798.6 | 665791.4 | False |  |  | 0 | 10 | False |
| floor_harsh_together_T30_20261001-180300 | 0.96 | 30 | 30 | 1317.718 | 1285 | 623 | 401163.1 | 662831.4 | False |  |  | 0 | 10 | False |
| floor_harsh_together_T5_20261001-184916 | 0.96 | 5 | 5 | 1316.991 | 1285 | 623 | 400928.6 | 662137.2 | False |  |  | 0 | 10 | False |
| floor_harsh_together_T5_20261002-150520 | 0.39 | 5 | 5 | 1214.949 | 520 | 0 | 6928.5 | 27690.5 | False |  |  | 0 | 10 | False |
| floor_harsh_together_T60_20261001-173952 | 0.96 | 60 | 60 | 1317.177 | 1301 | 640 | 400179.4 | 666261.1 | False |  |  | 0 | 10 | False |

Smallest clean setting: **2** s

Timeout floor (both workloads): **2** s

## 5. Verdict

- Case A: engine-watchdog T_detect median = 0.1 s -> **KILL**; any-component T_detect median = 0.1 s -> **KILL** (criteria: T_detect >= 60.0 s and floor 2 >= 10.0 s)
- Case B: engine-watchdog T_detect median = 351.9 s -> **KILL**; any-component T_detect median = 5.0 s -> **KILL** (criteria: T_detect >= 60.0 s and floor 2 >= 10.0 s)

## 6. Workarounds and observations

- harsh re-run at its own 70% point (0.39 req/s; saturation 0.55): watchdog 10/5/2 s all clean, 0 request errors, TTFT p50 ~7 s / p99 ~28 s. The 1-token /generate probe still hit up to 10 consecutive 5 s timeouts during 32k-prefill bursts, so a generation-based liveness probe is not usable as a router signal at this timeout without false positives; /health never left 200 because any output counts as healthy.
- Floor verdict: smallest clean watchdog setting is 2 s (lowest tested) in both workloads, including the overloaded harsh run. The stock default is 150x above the measured floor.


Harness-level workarounds baked in (each one is a finding about the stock stack):
- Per-rank logs and PIDs are not available from a stock launch (children inherit the parent's stdout); `launch_wrapped.py` redirects fds and records PIDs inside each spawned rank.
- Scheduler step boundaries are not exported; `run_batch`/`process_batch_result` are wrapped from the launcher to timestamp them.
- `/health` blocks up to 20 s server-side, so the probe keeps one request in flight per endpoint instead of polling synchronously at 100 ms.
- A SIGSTOPped rank must be SIGCONTed before it can be killed; `stop.py` does this before tearing the tree down.
- After the watchdog fires, the parent kills the tree; the load generator observes hung requests as connection resets, not as timeouts, so in-flight hang counts come from `send_ts < t_inject and error != None`.
