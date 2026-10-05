# Test 3: in-process failure containment (SGLang Diffusion, Wan SP=2 -> SP=1)

Timeline (s): T_detect = rank-1 freeze -> deadline miss on rank 0; T_switch = miss -> failed over (abort + coordinator shrink + peer kill); T_recompute = the failed step re-run at SP=1; T_added = failing request latency minus the SP=2 reference request in the same server.

## Recovery timeline

| run | shape | steps | fail_step | deadline_s | abort_mode | fence_first | abort_finished | T_detect_s | T_abort_s | T_switch_s | T_recompute_s | sp2_step_ms_median | sp1_step_ms_median | ref_latency_s | fail_latency_s | T_added_s | after_latency_s | sp1ref_latency_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| contain_ours_20261005-105418 | 832x480x81 | 9 | 4 | 5 | stuck | 1 | True | 5.002 | 0.440 | 1.596 | 0.501 | 716.676 | 1188.413 | 15.144 | 32.761 | 17.617 | 23.706 | 34.275 |
| contain_ours_20261005-105657 | 832x480x81 | 9 | 4 | 5 | stuck | 1 | True | 5.002 | 0.196 | 1.212 | 0.515 | 719.211 | 1198.961 | 15.145 | 32.775 | 17.629 | 24.210 | 34.275 |
| contain_ours_20261005-105937 | 832x480x81 | 9 | 4 | 5 | stuck | 1 | True | 5.002 | 0.088 | 1.246 | 0.510 | 720.464 | 1197.304 | 15.143 | 32.765 | 17.622 | 24.213 | 34.275 |

## Output check and state

| run | encoder_parallel | sharded_components | injected | rank1_after_fail | fail_ok | after_ok | fail_hash_eq_ref | after_hash_eq_ref | relerr_vs_sp2ref | relerr_vs_sp1ref | control_sp1_vs_sp2 | after_vs_sp2ref | after_vs_fail | after_vs_sp1ref | after_hash_eq_sp1ref | sp1ref_video_fhw | ref_video_fhw | fail_video_fhw | after_video_fhw | ref_video_mad_vs_sp1ref | fail_video_mad_vs_sp1ref | after_video_mad_vs_sp1ref | coordinators_shrunk | peers_killed | modules_sp_size_reset | errors | fail_error | after_error |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| contain_ours_20261005-105418 | replicate |  | True | Z | True | True | False | False | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False | 81x480x832 | 81x480x832 | 81x480x832 | 81x480x832 | 1.274 | 1.273 | 1.273 | _WORLD,_SP,_VAE_DECODE | [314795] | 1 |  |  |  |
| contain_ours_20261005-105657 | replicate |  | True | Z | True | True | False | False | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False | 81x480x832 | 81x480x832 | 81x480x832 | 81x480x832 | 1.274 | 1.273 | 1.273 | _WORLD,_SP,_VAE_DECODE | [317088] | 1 |  |  |  |
| contain_ours_20261005-105937 | replicate |  | True | Z | True | True | False | False | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False | 81x480x832 | 81x480x832 | 81x480x832 | 81x480x832 | 1.274 | 1.273 | 1.273 | _WORLD,_SP,_VAE_DECODE | [319367] | 1 |  |  |  |

Baselines measured earlier on this host (not re-run here): stock SGLang Diffusion never detects the frozen peer (Phase 2: 600.2 s to the torch watchdog, request lost, replica never evicted); kill -> cold restart of the rank is ~37 s to ready (startup_decomp.md) plus the lost request.
Verdict rule: PASS if the failing request completes with relerr_vs_sp1ref within the control (SP=1 vs SP=2 references) and the next request is served by the same process at SP=1 with after_vs_sp1ref also within the control (a fast but wrong next request is a FAIL). The decoded videos must also match: same frames x height x width as the SP=1 reference, and a mean abs pixel difference (0-255) no larger than the SP=2 reference's own.
