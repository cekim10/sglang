## Per run

| run | model | tp | watchdog_timeout | dist_timeout | rpc_timeout | stack | profile | rate | mode | case | target_rank | t_first_symptom | t_stall_engine | stall_engine_rank | t_target_state_seen | t_detect_engine | engine_detector | t_detect_health | health_state | t_detect_health_if_5s_client_timeout | t_detect_gen1 | t_detect_gen1_sustained3 | t_server_dead | n_health_false_pos_5min_before | n_gen1_false_pos_5min_before | window_end | n_inflight_hung | n_new_sent | n_new_errored | ttft_p50_before_ms | ttft_p99_before_ms | n_before | ttft_p50_during_ms | ttft_p99_during_ms | n_during_with_first_token | launch_failed | not_steady | no_engine_detect | no_self_teardown |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| diff_B_r1_1_20261002-221223 | Tongyi-MAI/Z-Image-Turbo | 2 | absent | default | default | sglang-diffusion | mixed | 0.376 | inject | B | 1 | -0.101 |  |  | 0.054 | 600.275 | log:nccl_timeout@rank0.log |  |  |  | 59.033 | 179.189 |  | 0 | 0 | 600.275 | 1 | 234 | 234 | 2636.1 | 8183.5 | 22 |  |  | 0 | False | False | False | True |
| diff_B_r1_2_20261002-225055 | Tongyi-MAI/Z-Image-Turbo | 2 | absent | default | default | sglang-diffusion | mixed | 0.376 | inject | B | 1 | -0.156 |  |  | 0.097 | 600.208 | log:nccl_timeout@rank0.log |  |  |  | 58.976 | 179.165 |  | 0 | 0 | 600.208 | 1 | 234 | 234 | 2591.0 | 8242.9 | 22 |  |  | 0 | False | False | False | True |
| diff_B_r1_3_20261002-232927 | Tongyi-MAI/Z-Image-Turbo | 2 | absent | default | default | sglang-diffusion | mixed | 0.376 | inject | B | 1 | -0.193 |  |  | 0.066 | 600.211 | log:nccl_timeout@rank0.log |  |  |  | 58.945 | 179.1 |  | 0 | 0 | 600.211 | 1 | 234 | 234 | 2494.9 | 8197.6 | 22 |  |  | 0 | False | False | False | True |

### T_detect per case (seconds after t_inject, median over runs)

| case | watchdog_timeout | runs | t_first_symptom_med | t_detect_engine_med | t_detect_engine_min | t_detect_engine_max | t_detect_health_med | t_detect_gen1_med | n_inflight_hung_med | n_new_errored_med |
|---|---|---|---|---|---|---|---|---|---|---|
| B | absent | 3 | -0.16 | 600.21 | 600.21 | 600.28 |  | 58.98 | 1.0 | 234.0 |
