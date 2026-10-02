## Per run

| run | model | tp | watchdog_timeout | dist_timeout | profile | rate | mode | case | target_rank | t_first_symptom | t_stall_engine | stall_engine_rank | t_target_state_seen | t_detect_engine | engine_detector | t_detect_health | health_state | t_detect_health_if_5s_client_timeout | t_detect_gen1 | t_detect_gen1_sustained3 | t_server_dead | n_health_false_pos_5min_before | n_gen1_false_pos_5min_before | window_end | n_inflight_hung | n_new_sent | n_new_errored | ttft_p50_before_ms | ttft_p99_before_ms | n_before | ttft_p50_during_ms | ttft_p99_during_ms | n_during_with_first_token | launch_failed | not_steady | no_engine_detect | no_self_teardown |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A_r1_1_20260929-231600 | Qwen/Qwen3-14B | 2 | default | default | mixed | 0.96 | inject | A | 1 | 0.011 | -0.011 | 0 | 0.102 | 0.103 | log:scheduler_exception@rank0.log | 59.081 | timeout | 4.078 | 5.004 | 15.173 | 65.35 | 0 | 1 | 0.103 | 3 | 0 | 0 | 447.3 | 5064.8 | 53 |  |  | 0 | False | False | False | False |
| A_r1_2_20260929-235034 | Qwen/Qwen3-14B | 2 | default | default | mixed | 0.96 | inject | A | 1 | 0.014 | -0.008 | 0 | 0.007 | 0.11 | log:scheduler_exception@rank0.log | 59.085 | timeout | 4.082 | 5.009 | 15.152 | 65.336 | 0 | 1 | 0.11 | 3 | 0 | 0 | 421.6 | 5057.0 | 53 |  |  | 0 | False | False | False | False |
| A_r1_3_20260930-002508 | Qwen/Qwen3-14B | 2 | default | default | mixed | 0.96 | inject | A | 1 | 0.003 | -0.02 | 0 | 0.024 | 0.024 | log:scheduler_exception@rank0.log | 5.132 | conn_error:RemoteProtocolError | 4.103 | 5.026 | 5.177 | 5.175 | 0 | 1 | 0.024 | 3 | 0 | 0 | 442.4 | 5041.0 | 53 |  |  | 0 | False | False | False | False |
| A_r1_4_20260930-005942 | Qwen/Qwen3-14B | 2 | default | default | mixed | 0.96 | inject | A | 1 | 0.016 | -0.006 | 0 | 0.041 | 0.042 | log:scheduler_exception@rank0.log | 59.124 | timeout | 4.121 | 5.044 | 15.192 | 65.286 | 0 | 1 | 0.042 | 3 | 0 | 0 | 421.7 | 5054.0 | 53 |  |  | 0 | False | False | False | False |
| A_r1_5_20260930-013416 | Qwen/Qwen3-14B | 2 | default | default | mixed | 0.96 | inject | A | 1 | 0.018 | -0.005 | 0 | 0.012 | 0.115 | log:scheduler_exception@rank0.log | 59.093 | timeout | 4.09 | 5.015 | 15.175 | 65.296 | 0 | 1 | 0.115 | 3 | 0 | 0 | 433.4 | 5041.5 | 53 |  |  | 0 | False | False | False | False |

### T_detect per case (seconds after t_inject, median over runs)

| case | watchdog_timeout | runs | t_first_symptom_med | t_detect_engine_med | t_detect_engine_min | t_detect_engine_max | t_detect_health_med | t_detect_gen1_med | n_inflight_hung_med | n_new_errored_med |
|---|---|---|---|---|---|---|---|---|---|---|
| A | default | 5 | 0.01 | 0.1 | 0.02 | 0.12 | 59.08 | 5.01 | 3.0 | 0.0 |
