## Per run

| run | model | tp | watchdog_timeout | dist_timeout | profile | rate | mode | duration_s | n_requests | n_errors | error_kinds | ttft_p50_ms | ttft_p99_ms | spurious_kill | first_engine_event | first_pid_death | health_non200_changes | gen1_non200_changes | gen1_max_consecutive_bad | launch_failed | not_steady | no_engine_detect | no_self_teardown |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| floor_harsh_together_T10_20261001-182608 | Qwen/Qwen3-14B | 2 | 10 | 10 | harsh | 0.96 | floor | 1316.974 | 1285 | 623 | ReadError: ; cancelled: load generator stopped while  | 400834.4 | 662099.0 | False |  |  | 0 | 1 | 10 | False | False | False | False |
| floor_harsh_together_T10_20261002-144357 | Qwen/Qwen3-14B | 2 | 10 | 10 | harsh | 0.39 | floor | 1214.931 | 520 | 0 |  | 6956.9 | 27566.2 | False |  |  | 0 | 26 | 10 | False | False | False | False |
| floor_harsh_together_T2_20261001-191225 | Qwen/Qwen3-14B | 2 | 2 | 2 | harsh | 0.96 | floor | 1320.075 | 1285 | 624 | ReadError: ; cancelled: load generator stopped while  | 403224.8 | 666396.5 | False |  |  | 0 | 1 | 10 | False | False | False | False |
| floor_harsh_together_T2_20261002-152643 | Qwen/Qwen3-14B | 2 | 2 | 2 | harsh | 0.39 | floor | 1214.937 | 520 | 0 |  | 7007.8 | 27670.4 | False |  |  | 0 | 26 | 10 | False | False | False | False |
| floor_harsh_together_T300_20261001-171643 | Qwen/Qwen3-14B | 2 | 300 | 300 | harsh | 0.96 | floor | 1316.707 | 1301 | 640 | ReadError: ; cancelled: load generator stopped while  | 399798.6 | 665791.4 | False |  |  | 0 | 1 | 10 | False | False | False | False |
| floor_harsh_together_T30_20261001-180300 | Qwen/Qwen3-14B | 2 | 30 | 30 | harsh | 0.96 | floor | 1317.718 | 1285 | 623 | ReadError: ; cancelled: load generator stopped while  | 401163.1 | 662831.4 | False |  |  | 0 | 1 | 10 | False | False | False | False |
| floor_harsh_together_T5_20261001-184916 | Qwen/Qwen3-14B | 2 | 5 | 5 | harsh | 0.96 | floor | 1316.991 | 1285 | 623 | ReadError: ; cancelled: load generator stopped while  | 400928.6 | 662137.2 | False |  |  | 0 | 1 | 10 | False | False | False | False |
| floor_harsh_together_T5_20261002-150520 | Qwen/Qwen3-14B | 2 | 5 | 5 | harsh | 0.39 | floor | 1214.949 | 520 | 0 |  | 6928.5 | 27690.5 | False |  |  | 0 | 30 | 10 | False | False | False | False |
| floor_harsh_together_T60_20261001-173952 | Qwen/Qwen3-14B | 2 | 60 | 60 | harsh | 0.96 | floor | 1317.177 | 1301 | 640 | ReadError: ; cancelled: load generator stopped while  | 400179.4 | 666261.1 | False |  |  | 0 | 1 | 10 | False | False | False | False |

### Timeout floor sweep (spurious kills per setting)

| profile | rate | watchdog_timeout | dist_timeout | runs | spurious | errors | ttft_p99_ms |
|---|---|---|---|---|---|---|---|
| harsh | 0.39 | 10 | 10 | 1 | 0 | 0 | 27566.2 |
| harsh | 0.39 | 2 | 2 | 1 | 0 | 0 | 27670.4 |
| harsh | 0.39 | 5 | 5 | 1 | 0 | 0 | 27690.5 |
| harsh | 0.96 | 10 | 10 | 1 | 0 | 623 | 662099.0 |
| harsh | 0.96 | 2 | 2 | 1 | 0 | 624 | 666396.5 |
| harsh | 0.96 | 30 | 30 | 1 | 0 | 623 | 662831.4 |
| harsh | 0.96 | 300 | 300 | 1 | 0 | 640 | 665791.4 |
| harsh | 0.96 | 5 | 5 | 1 | 0 | 623 | 662137.2 |
| harsh | 0.96 | 60 | 60 | 1 | 0 | 640 | 666261.1 |
