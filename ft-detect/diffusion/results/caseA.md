## Per run

| run | model | tp | watchdog_timeout | dist_timeout | rpc_timeout | stack | profile | rate | mode | case | target_rank | t_first_symptom | t_stall_engine | stall_engine_rank | t_target_state_seen | t_detect_engine | engine_detector | t_detect_health | health_state | t_detect_health_if_5s_client_timeout | t_detect_gen1 | t_detect_gen1_sustained3 | t_server_dead | n_health_false_pos_5min_before | n_gen1_false_pos_5min_before | window_end | n_inflight_hung | n_new_sent | n_new_errored | ttft_p50_before_ms | ttft_p99_before_ms | n_before | ttft_p50_during_ms | ttft_p99_during_ms | n_during_with_first_token | launch_failed | not_steady | no_engine_detect | no_self_teardown |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| diff_A_r1_1_20261002-203027 | Tongyi-MAI/Z-Image-Turbo | 2 | absent | default | default | sglang-diffusion | mixed | 0.376 | inject | A | 1 | -0.209 |  |  | 0.046 | 600.23 | log:nccl_timeout@rank0.log |  |  |  | 58.925 | 179.0 |  | 0 | 0 | 600.23 | 0 | 235 | 235 | 2491.7 | 8161.6 | 22 |  |  | 0 | False | False | False | True |
| diff_A_r1_1_20261002-210331 | Tongyi-MAI/Z-Image-Turbo | 2 | absent | default | default | sglang-diffusion | mixed | 0.376 | inject | A | 1 | -0.156 |  |  | 0.102 | 600.217 | log:nccl_timeout@rank0.log |  |  |  | 58.98 | 179.12 |  | 0 | 0 | 600.217 | 0 | 235 | 235 | 2596.1 | 8228.6 | 22 |  |  | 0 | False | False | False | True |
| diff_A_r1_2_20261002-213757 | Tongyi-MAI/Z-Image-Turbo | 2 | absent | default | default | sglang-diffusion | mixed | 0.376 | inject | A | 1 | -0.189 |  |  | 0.07 | 600.246 | log:nccl_timeout@rank0.log |  |  |  | 58.947 | 179.1 |  | 0 | 0 | 600.246 | 1 | 234 | 234 | 2509.4 | 8175.4 | 22 |  |  | 0 | False | False | False | True |

### T_detect per case (seconds after t_inject, median over runs)

| case | watchdog_timeout | runs | t_first_symptom_med | t_detect_engine_med | t_detect_engine_min | t_detect_engine_max | t_detect_health_med | t_detect_gen1_med | n_inflight_hung_med | n_new_errored_med |
|---|---|---|---|---|---|---|---|---|---|---|
| A | absent | 3 | -0.19 | 600.23 | 600.22 | 600.25 |  | 58.95 | 0.0 | 235.0 |
