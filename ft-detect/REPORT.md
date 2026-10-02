# Failure-detection latency kill test — Phase 1 (SGLang, TP=2)

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
- The scheduler watchdog counts `forward_ct`, polls every `timeout/2`, so a hang is noticed 1.0x-1.5x the timeout after the last step, then sleeps 5 s and SIGQUITs the parent. The parent's SIGQUIT handler sleeps 5 s, runs py-spy, waits `SGLANG_CUDA_COREDUMP_BEFORE_CRASH_WAIT_SECS` (60 s, even when coredumps are not enabled) and only then kills the tree. The handler is synchronous in the main thread, so the HTTP event loop is frozen while it runs. The watchdog path has no 'unhealthy' state of its own; the only such state is set by `/health` after 20 s of silence, and nothing in the engine acts on it.
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

Median over runs (`teardown` = HTTP server process gone, i.e. the replica actually left service):

| case | watchdog_timeout | runs | engine_med | engine_min | engine_max | health_med | health_5s_client_med | gen1_med | teardown_med | teardown_min | teardown_max | hung_med | new_err_med |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | default | 5 | 0.1 | 0.02 | 0.12 | 59.08 | 4.09 | 5.01 | 65.3 | 5.18 | 65.35 | 3.0 | 0.0 |
| B | default | 8 | 351.95 | 305.51 | 395.31 | 20.15 | 5.14 | 4.96 | 422.23 | 375.71 | 465.6 | 3.0 | 338.0 |

Column meanings: `engine` = first detector string in any log or death of a non-injected process; `health` = first non-200 from /health or /health_generate with the probe's 60 s client timeout; `health_5s_client` = when a checker with a 5 s client timeout would first have flagged it (derived from the same probes); `gen1` = 1-token /generate probe, 5 s client timeout.

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

Criteria as specified: GO if T_detect at stock defaults >= 60 s AND the zero-false-positive timeout floor >= 10 s. Measured floor (both workloads): **2 s**, which is the lowest value tested, so the floor condition **fails regardless of T_detect**.

| case | runs | T_detect engine (median s) | T_detect any component (median s) | T_evict: replica gone (median s) | verdict, engine reading | verdict, any-component reading |
|---|---|---|---|---|---|---|
| A | 5 | 0.1 | 0.1 | 65.3 | KILL | KILL |
| B | 8 | 351.9 | 5.0 | 422.2 | KILL | KILL |

**Verdict: KILL** for the hypothesis as stated. The stock watchdog does take ~6 min to notice a hung rank, but that number is a default, not a floor: the same watchdog ran with zero false positives at 2 s under both workloads, including an overloaded one, because it measures per-step duration rather than request latency.

What the data does support (candidate Phase 2 framing, not claimed here): the interval that stays long after the timeout is lowered is detect-to-evict. After the engine has noticed a failure, the stock crash handler keeps the replica's port open and its HTTP loop frozen for about 65 s more (5 s settle + py-spy + 60 s coredump wait), so neither /health nor a router can learn anything until the process dies; with a 2 s watchdog a hang would still take ~72 s to leave service. Two further blind spots were observed in passing: a startup hang in distributed init that no component detected for 6 h 38 min, and liveness probes whose semantics make them either blind to overload (/health: any output counts) or false-positive-prone under it (1-token generate).

## 6. Workarounds and observations

Observations recorded during Phase 1 on elves-01 (2x L40S, SGLang 0.5.19, TP=2). Each item is something the stock stack did that the harness had to work around or that bears directly on detection latency.

- Environment deviation: the task assumed one node with 8 GPUs and --tp 4; the hosts have 2 GPUs each, so all runs use --tp 2 on one node. The process tree, watchdog, SubprocessWatchdog and /health paths are identical to the TP=4 case; only the number of ranks differs.
- Install: current SGLang releases require CUDA 13 and the host driver supports 12.9, so the last CUDA 12 release (0.5.19, torch 2.13.0+cu129, NCCL 2.29.7) was installed per the v0.5.19 install docs. The defaults measured (watchdog 300 s, /health 20 s, crash-handler waits) are unchanged on current main (c7be3e935b).
- elves-01 topology: GPU0 and GPU1 sit on different NUMA nodes (nvidia-smi topo = SYS). NCCL picks P2P/CUMEM by default and the transfer silently moves no data (a 2-rank all_reduce returned each rank's own value until torch's 60 s PG watchdog fired). Workaround for every run: NCCL_P2P_DISABLE=1 plus --disable-custom-all-reduce (SHM path, 4.3 ms per 64 MB all-reduce). This does not change the detector picture: TP all-reduce still runs through pynccl, outside torch's ProcessGroup watchdog.
- Startup hang, undetected for 6 h 38 min: with stock settings the server stalled in "Init torch distributed begin" (custom all-reduce IPC handshake over the broken P2P path). Nothing noticed: the scheduler watchdog is created after model init, /health was connection-refused, and torch's 600 s NCCL default never applied because the stuck collective was not a torch ProcessGroup op. Initialization hangs are outside every detector measured here.
- Case A (kill -9 rank 1): the surviving rank noticed in 0.02-0.12 s in 5/5 runs via "Scheduler hit an exception" (gloo broadcast peer reset). In one earlier run (discarded, old harness) it was instead the 1 s SubprocessWatchdog poll at 1.47 s, because rank 0 was inside an NCCL collective at the time; which detector fires depends on which collective the survivor is in.
- Case A teardown is bimodal: 65.3 s in 4/5 runs, 5.2 s in 1/5. The SIGQUIT handler sleeps 5 s, tries py-spy (ptrace denied on this host), then waits 60 s for CUDA coredumps that are not enabled (SGLANG_CUDA_COREDUMP_BEFORE_CRASH defaults True, WAIT_SECS 60). If rank 0 has already exited when the handler looks for scheduler processes it skips straight to kill_process_tree, hence the 5 s mode. The handler is synchronous in the main thread, so the HTTP event loop is frozen the whole time: /health cannot even return 503; our 60 s client timeout is what "detected" it, and a 5 s client timeout would have at ~4 s.
- Case B (SIGSTOP rank 1): watchdog fired at 351-352 s in 5/5 fixed-phase runs and at 305, 382, 395 s in 3 jittered runs, consistent with the 1.0x-1.5x window from the timeout/2 poll. /health returned 503 at 20.1-21.2 s in all 8 runs; the 1-token probe failed at its 5 s timeout in all 8. Replica left service 70 s after the watchdog fired (375-466 s). Every request admitted during the window was lost (294-373 per run at 0.96 req/s). SGLang takes no action on its own UnHealthy state; the port stays open until the tree is killed.
- Health probe false positives under normal load: in the first (discarded) case A run /health returned 503 twice and the 1-token probe timed out repeatedly with no fault injected, during a cold-start burst of long prefills (40 s with no decode output). In the 13 kept injection runs, /health false positives in the 5 min before injection were 0 and 1-token false positives were 0-1 per run.
- Floor sweep, mixed at 0.96 req/s: zero watchdog false positives at 300/60/30/10/5/2 s; dist_timeout 2 s also started fine (48 s). The watchdog measures per-step duration, not request latency, so queueing cannot trip it while 4096-token prefill chunks finish in < 2 s on this hardware.
- Floor sweep, harsh at 0.96 req/s (the mixed rate; an overload for this profile: TTFT p50 ~400 s, ~half the requests failed/cancelled): still zero watchdog false positives; /health stayed 200 throughout because any output counts as healthy; the 1-token probe showed up to 10 consecutive 5 s timeouts.
- harsh re-run at its own 70% point (0.39 req/s; saturation 0.55): watchdog 10/5/2 s all clean, 0 request errors, TTFT p50 ~7 s / p99 ~28 s. The 1-token /generate probe still hit up to 10 consecutive 5 s timeouts during 32k-prefill bursts, so a generation-based liveness probe is not usable as a router signal at this timeout without false positives; /health never left 200 because any output counts as healthy.
- Floor verdict: smallest clean watchdog setting is 2 s (lowest tested) in both workloads, including the overloaded harsh run. The stock default is 150x above the measured floor. With a 2 s watchdog a Case B hang would still take ~72 s to leave service (2 + 5 + 65 s of crash-handler waits).
- Measurement caveat: a fixed launch-to-inject schedule samples the same watchdog poll phase every run (hence 351-352 s five times); INJECT_JITTER was added and the 3 jittered runs span 305-395 s. Case A runs were injected at steady state with ~3 requests in flight at 0.96 req/s, so in-flight hang counts are small by construction.


Harness-level workarounds baked in (each one is a finding about the stock stack):
- Per-rank logs and PIDs are not available from a stock launch (children inherit the parent's stdout); `launch_wrapped.py` redirects fds and records PIDs inside each spawned rank.
- Scheduler step boundaries are not exported; `run_batch`/`process_batch_result` are wrapped from the launcher to timestamp them.
- `/health` blocks up to 20 s server-side, so the probe keeps one request in flight per endpoint instead of polling synchronously at 100 ms.
- A SIGSTOPped rank must be SIGCONTed before it can be killed; `stop.py` does this before tearing the tree down.
- After the watchdog fires, the parent kills the tree; the load generator observes hung requests as connection resets, not as timeouts, so in-flight hang counts come from `send_ts < t_inject and error != None`.
