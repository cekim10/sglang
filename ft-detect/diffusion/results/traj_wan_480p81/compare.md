Trajectory portability: {'tag': 'wan_480p81', 'model': 'Wan-AI/Wan2.2-TI2V-5B-Diffusers', 'shape': '832x480x81', 'steps': 9, 'k': 3, 'seed': 1234}

| config | ok | exact_vs_sp1_ref | max_abs | rel_l2 | mean_abs | image_sha_equal_sp1_ref | latency_s | steps_executed | first_step_ms | warm_step_ms | t_save_ms | save_bytes | t_restore_ms | t_first_resumed_step_ms | t_remaining_ms | restore_notes | error |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sp2_save | True | False | 2.244e+00 | 9.017e-02 | 8.246e-02 | False | 28.71 | 9 | 2269.1 | 700.0 | 344.3 | 25163398 |  |  |  |  |  |
| sp1_ref | True | True | 0.000e+00 | 0.000e+00 | 0.000e+00 | True | 47.31 | 9 | 3259.4 | 1198.3 |  |  |  |  |  |  |  |
| sp1_ref2 | True | True | 0.000e+00 | 0.000e+00 | 0.000e+00 | True | 46.3 | 9 | 3231.3 | 1198.7 |  |  |  |  |  |  |  |
| sp1_full | True | False | 2.339e+00 | 8.986e-02 | 8.258e-02 | False | 41.33 | 5 | 3218.4 | 1165.4 |  |  | 17.8 | 3218.6 | 7927.2 |  |  |
| sp1_lower | True | False | 2.357e+00 | 1.032e-01 | 9.529e-02 | False | 41.32 | 5 | 3228.9 | 1198.5 |  |  | 15.8 | 3229.1 | 8056.1 |  |  |

Reading: `sp1_ref2` = run-to-run noise; `sp2_save` vs `sp1_ref` = SP2<->SP1 control (collective ordering); `sp1_full` = resumed with the full solver history (the gate: should match sp1_ref to within the control); `sp1_lower` = resumed with history reset (quality cost of a low-order restart).
