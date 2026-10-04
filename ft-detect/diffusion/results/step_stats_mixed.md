Denoising-step durations by shape over 3 run(s); floor = max observed step (or inter-step gap) x 2 safety.

| shape | pixels_frames | steps | requests | step_p50_ms | step_p99_ms | step_max_ms | gap_max_ms | outliers_gt5x_p50 | max_is_cold_shape | step_max_warm_ms | per_shape_floor_s | per_shape_floor_warm_s | slack_vs_global_x |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 256x256 | 65536 | 3048 | 1524 | 43.2 | 47.4 | 253.5 |  | 2 | True | 57.6 | 0.51 | 0.12 | 2.1 |
| 512x512 | 262144 | 8316 | 462 | 66.2 | 68.5 | 72.0 | 72.1 | 0 | False | 72.0 | 0.14 | 0.14 | 7.6 |
| 1024x1024 | 1048576 | 2916 | 162 | 222.5 | 225.1 | 230.2 | 230.5 | 0 | True | 230.0 | 0.46 | 0.46 | 2.3 |
| 1536x1536 | 2359296 | 1188 | 66 | 532.9 | 536.3 | 536.7 | 536.8 | 0 | False | 536.7 | 1.07 | 1.07 | 1.0 |

**Global step-watchdog floor (one timeout for every served shape): 1.07 s (all steps) / 1.07 s (excluding each shape's first request, i.e. cold-shape warm-up).** Smallest per-shape floor: 0.14 s (a single global timeout is 8x slack on that shape). `max_is_cold_shape` says whether a shape's slowest step was its very first request.
