Execution-quantum durations over 1 run(s). D_w_step = 2 x Q0.999(warm step); D_w_stage = 2 x max warm non-denoising stage (e.g. VAE decode); D_w = max of the two; D_global = max_w D_w.

| model | shape | pixels_frames | steps | requests | step_p50_ms | step_p99_ms | step_p999_ms | step_max_ms | gap_max_ms | warm_steps | warm_q_ms | warm_max_ms | max_first_after_launch_ms | max_first_of_shape_ms | max_after_idle_ms | D_w_step_s | D_w_all_s | stage_warm_max_ms | stage_warm_name | stage_cold_max_ms | D_w_stage_s | D_w_s | D_w_quantum | slack_x |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Tongyi-MAI/Z-Image-Turbo | 256x256 | 65536 | 108 | 6 | 42.5 | 198.3 | 267.5 | 274.5 | 46.1 | 72 | 45.5 | 45.6 | 274.5 |  | 47.7 | 0.091 | 0.549 | 273.8 | TextEncodingStage | 1713.3 | 0.548 | 0.548 | TextEncodingStage | 3.3 |
| Tongyi-MAI/Z-Image-Turbo | 512x512 | 262144 | 108 | 6 | 65.0 | 72.2 | 72.7 | 72.8 | 73.0 | 90 | 72.5 | 72.8 |  | 72.4 |  | 0.145 | 0.146 | 267.4 | TextEncodingStage |  | 0.535 | 0.535 | TextEncodingStage | 3.4 |
| Tongyi-MAI/Z-Image-Turbo | 1024x1024 | 1048576 | 108 | 6 | 218.9 | 221.1 | 221.3 | 221.3 | 221.4 | 90 | 221.3 | 221.3 |  | 220.4 |  | 0.443 | 0.443 | 323.3 | TextEncodingStage |  | 0.647 | 0.647 | TextEncodingStage | 2.8 |
| Tongyi-MAI/Z-Image-Turbo | 1536x1536 | 2359296 | 108 | 6 | 529.2 | 532.2 | 532.4 | 532.4 | 532.6 | 90 | 532.4 | 532.4 |  | 527.8 |  | 1.065 | 1.065 | 913.3 | decoding_stage |  | 1.827 | 1.827 | decoding_stage | 1.0 |

**D_global = 1.83 s (set by 1536x1536 / decoding_stage); smallest D_w = 0.54 s; max slack = 3x.** Cold categories: `max_first_after_launch_ms`, `max_first_of_shape_ms`, `max_after_idle_ms` are the slowest step of such requests (a static watchdog must tolerate them; a progress-aware one can know they are coming).

Non-denoising stages (single execution quanta without step boundaries):

| model | shape | stage | n | p50_ms | max_ms |
|---|---|---|---|---|---|
| Tongyi-MAI/Z-Image-Turbo | 1024x1024 | InputValidationStage | 12 | 0.1 | 0.2 |
| Tongyi-MAI/Z-Image-Turbo | 1024x1024 | LatentPreparationStage | 12 | 5.9 | 6.0 |
| Tongyi-MAI/Z-Image-Turbo | 1024x1024 | TextEncodingStage | 12 | 319.9 | 323.3 |
| Tongyi-MAI/Z-Image-Turbo | 1024x1024 | TimestepPreparationStage | 12 | 0.5 | 0.6 |
| Tongyi-MAI/Z-Image-Turbo | 1024x1024 | decoding_stage | 12 | 54.5 | 55.6 |
| Tongyi-MAI/Z-Image-Turbo | 1536x1536 | InputValidationStage | 12 | 0.1 | 0.2 |
| Tongyi-MAI/Z-Image-Turbo | 1536x1536 | LatentPreparationStage | 12 | 5.9 | 6.0 |
| Tongyi-MAI/Z-Image-Turbo | 1536x1536 | TextEncodingStage | 12 | 319.6 | 322.3 |
| Tongyi-MAI/Z-Image-Turbo | 1536x1536 | TimestepPreparationStage | 12 | 0.5 | 0.6 |
| Tongyi-MAI/Z-Image-Turbo | 1536x1536 | decoding_stage | 12 | 907.3 | 913.3 |
| Tongyi-MAI/Z-Image-Turbo | 256x256 | InputValidationStage | 12 | 0.2 | 0.6 |
| Tongyi-MAI/Z-Image-Turbo | 256x256 | LatentPreparationStage | 12 | 5.8 | 6.0 |
| Tongyi-MAI/Z-Image-Turbo | 256x256 | TextEncodingStage | 12 | 267.0 | 1713.3 |
| Tongyi-MAI/Z-Image-Turbo | 256x256 | TimestepPreparationStage | 12 | 0.5 | 0.7 |
| Tongyi-MAI/Z-Image-Turbo | 256x256 | decoding_stage | 12 | 7.0 | 159.8 |
| Tongyi-MAI/Z-Image-Turbo | 512x512 | InputValidationStage | 12 | 0.1 | 0.2 |
| Tongyi-MAI/Z-Image-Turbo | 512x512 | LatentPreparationStage | 12 | 5.9 | 6.0 |
| Tongyi-MAI/Z-Image-Turbo | 512x512 | TextEncodingStage | 12 | 266.9 | 267.4 |
| Tongyi-MAI/Z-Image-Turbo | 512x512 | TimestepPreparationStage | 12 | 0.5 | 0.6 |
| Tongyi-MAI/Z-Image-Turbo | 512x512 | decoding_stage | 12 | 8.2 | 41.3 |
