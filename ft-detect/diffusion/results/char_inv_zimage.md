Execution-quantum durations over 1 run(s). D_w_step = 2 x Q0.999(warm step); D_w_stage = 2 x max warm non-denoising stage (e.g. VAE decode); D_w = max of the two; D_global = max_w D_w.

| model | shape | pixels_frames | steps | requests | step_p50_ms | step_p99_ms | step_p999_ms | step_max_ms | gap_max_ms | warm_steps | warm_q_ms | warm_max_ms | max_first_after_launch_ms | max_first_of_shape_ms | max_after_idle_ms | D_w_step_s | D_w_all_s | stage_warm_max_ms | stage_warm_name | stage_cold_max_ms | D_w_stage_s | D_w_s | D_w_quantum | slack_x |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Tongyi-MAI/Z-Image-Turbo | 512x512 | 262144 | 36 | 2 | 63.9 | 239.9 | 241.1 | 241.2 | 70.5 | 18 | 66.6 | 66.6 | 241.2 |  |  | 0.133 | 0.482 | 267.8 | TextEncodingStage | 1648.9 | 0.536 | 0.536 | TextEncodingStage | 3.4 |
| Tongyi-MAI/Z-Image-Turbo | 1536x1536 | 2359296 | 36 | 2 | 526.3 | 528.4 | 528.4 | 528.4 | 530.1 | 18 | 527.6 | 527.6 |  | 528.4 |  | 1.055 | 1.06 | 906.2 | decoding_stage |  | 1.812 | 1.812 | decoding_stage | 1.0 |

**D_global = 1.81 s (set by 1536x1536 / decoding_stage); smallest D_w = 0.54 s; max slack = 3x.** Cold categories: `max_first_after_launch_ms`, `max_first_of_shape_ms`, `max_after_idle_ms` are the slowest step of such requests (a static watchdog must tolerate them; a progress-aware one can know they are coming).

Non-denoising stages (single execution quanta without step boundaries):

| model | shape | stage | n | p50_ms | max_ms |
|---|---|---|---|---|---|
| Tongyi-MAI/Z-Image-Turbo | 1536x1536 | InputValidationStage | 4 | 0.1 | 0.2 |
| Tongyi-MAI/Z-Image-Turbo | 1536x1536 | LatentPreparationStage | 4 | 5.9 | 5.9 |
| Tongyi-MAI/Z-Image-Turbo | 1536x1536 | TextEncodingStage | 4 | 288.2 | 323.9 |
| Tongyi-MAI/Z-Image-Turbo | 1536x1536 | TimestepPreparationStage | 4 | 0.5 | 0.5 |
| Tongyi-MAI/Z-Image-Turbo | 1536x1536 | decoding_stage | 4 | 902.9 | 906.2 |
| Tongyi-MAI/Z-Image-Turbo | 512x512 | InputValidationStage | 4 | 0.4 | 0.9 |
| Tongyi-MAI/Z-Image-Turbo | 512x512 | LatentPreparationStage | 4 | 3.4 | 6.0 |
| Tongyi-MAI/Z-Image-Turbo | 512x512 | TextEncodingStage | 4 | 956.8 | 1648.9 |
| Tongyi-MAI/Z-Image-Turbo | 512x512 | TimestepPreparationStage | 4 | 0.6 | 0.8 |
| Tongyi-MAI/Z-Image-Turbo | 512x512 | decoding_stage | 4 | 67.8 | 137.6 |
