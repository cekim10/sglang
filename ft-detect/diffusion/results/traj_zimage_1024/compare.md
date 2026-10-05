Trajectory portability: {'tag': 'zimage_1024', 'model': 'Tongyi-MAI/Z-Image-Turbo', 'shape': '1024x1024', 'steps': 9, 'k': 3, 'seed': 1234}

| config | ok | exact_vs_sp1_ref | max_abs | rel_l2 | mean_abs | image_sha_equal_sp1_ref | latency_s | steps_executed | first_step_ms | warm_step_ms | t_save_ms | save_bytes | t_restore_ms | t_first_resumed_step_ms | t_remaining_ms | restore_notes | error |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sp2_save | True | False | 8.281e-01 | 2.042e-02 | 1.346e-02 | False | 4.56 | 9 | 341.2 | 221.5 | 170.4 | 526573 |  |  |  |  |  |
| sp1_ref | True | True | 0.000e+00 | 0.000e+00 | 0.000e+00 | True | 5.2 | 9 | 357.7 | 295.0 |  |  |  |  |  |  |  |
| sp1_ref2 | True | True | 0.000e+00 | 0.000e+00 | 0.000e+00 | True | 5.1 | 9 | 350.5 | 293.3 |  |  |  |  |  |  |  |
| sp1_full | True | False | 8.633e-01 | 2.038e-02 | 1.355e-02 | False | 3.83 | 5 | 340.7 | 290.3 |  |  | 2.3 | 340.9 | 1520.9 |  |  |
| sp1_lower | True | False | 8.633e-01 | 2.038e-02 | 1.355e-02 | False | 3.95 | 5 | 355.4 | 292.2 |  |  | 2.4 | 355.5 | 1535.6 |  |  |

Reading: `sp1_ref2` = run-to-run noise; `sp2_save` vs `sp1_ref` = SP2<->SP1 control (collective ordering); `sp1_full` = resumed with the full solver history (the gate: should match sp1_ref to within the control); `sp1_lower` = resumed with history reset (quality cost of a low-order restart).
