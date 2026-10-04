Execution-quantum durations over 1 run(s). D_w_step = 2 x Q0.999(warm step); D_w_stage = 2 x max warm non-denoising stage (e.g. VAE decode); D_w = max of the two; D_global = max_w D_w.

| model | shape | pixels_frames | steps | requests | step_p50_ms | step_p99_ms | step_p999_ms | step_max_ms | gap_max_ms | warm_steps | warm_q_ms | warm_max_ms | max_first_after_launch_ms | max_first_of_shape_ms | max_after_idle_ms | D_w_step_s | D_w_all_s | stage_warm_max_ms | stage_warm_name | stage_cold_max_ms | D_w_stage_s | D_w_s | D_w_quantum | slack_x |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 512x512 | 262144 | 36 | 2 | 84.7 | 2425.0 | 2687.3 | 2716.5 | 109.3 | 18 | 88.7 | 88.7 | 2716.5 |  |  | 0.177 | 5.433 | 431.1 | TextEncodingStage | 2325.9 | 0.862 | 0.862 | TextEncodingStage | 29.7 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 832x480x17f | 6789120 | 36 | 2 | 167.4 | 171.9 | 171.9 | 171.9 | 184.8 | 18 | 171.9 | 171.9 |  | 171.6 |  | 0.344 | 0.37 | 1219.3 | decoding_stage |  | 2.439 | 2.439 | decoding_stage | 10.5 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 832x480x81f | 32348160 | 36 | 2 | 705.0 | 727.3 | 727.4 | 727.4 | 756.8 | 18 | 727.4 | 727.4 |  | 722.4 |  | 1.455 | 1.514 | 5973.1 | decoding_stage |  | 11.946 | 11.946 | decoding_stage | 2.1 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 832x480x85f | 33945600 | 36 | 2 | 765.9 | 789.2 | 789.3 | 789.3 | 822.0 | 18 | 789.3 | 789.3 |  | 786.8 |  | 1.579 | 1.644 | 6253.0 | decoding_stage |  | 12.506 | 12.506 | decoding_stage | 2.0 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 1280x704x81f | 72990720 | 36 | 2 | 2034.7 | 2095.0 | 2095.0 | 2095.0 | 2152.4 | 18 | 2095.0 | 2095.0 |  | 2092.2 |  | 4.19 | 4.305 | 12780.5 | decoding_stage |  | 25.561 | 25.561 | decoding_stage | 1.0 |

**D_global = 25.56 s (set by 1280x704x81f / decoding_stage); smallest D_w = 0.86 s; max slack = 30x.** Cold categories: `max_first_after_launch_ms`, `max_first_of_shape_ms`, `max_after_idle_ms` are the slowest step of such requests (a static watchdog must tolerate them; a progress-aware one can know they are coming).

Non-denoising stages (single execution quanta without step boundaries):

| model | shape | stage | n | p50_ms | max_ms |
|---|---|---|---|---|---|
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 1280x704x81f | InputValidationStage | 4 | 0.4 | 0.6 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 1280x704x81f | LatentPreparationStage | 4 | 1.9 | 2.3 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 1280x704x81f | TextEncodingStage | 4 | 438.8 | 442.3 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 1280x704x81f | TimestepPreparationStage | 4 | 0.6 | 0.7 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 1280x704x81f | decoding_stage | 4 | 12778.2 | 12780.5 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 512x512 | InputValidationStage | 4 | 0.4 | 0.5 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 512x512 | LatentPreparationStage | 4 | 1.5 | 1.8 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 512x512 | TextEncodingStage | 4 | 1376.0 | 2325.9 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 512x512 | TimestepPreparationStage | 4 | 0.6 | 0.7 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 512x512 | decoding_stage | 4 | 151.1 | 278.8 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 832x480x17f | InputValidationStage | 4 | 0.4 | 0.5 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 832x480x17f | LatentPreparationStage | 4 | 1.7 | 1.9 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 832x480x17f | TextEncodingStage | 4 | 435.7 | 441.6 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 832x480x17f | TimestepPreparationStage | 4 | 0.6 | 0.6 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 832x480x17f | decoding_stage | 4 | 1204.2 | 1219.3 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 832x480x81f | InputValidationStage | 4 | 0.4 | 0.6 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 832x480x81f | LatentPreparationStage | 4 | 1.7 | 2.0 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 832x480x81f | TextEncodingStage | 4 | 438.3 | 441.7 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 832x480x81f | TimestepPreparationStage | 4 | 0.5 | 0.7 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 832x480x81f | decoding_stage | 4 | 5957.3 | 5973.1 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 832x480x85f | InputValidationStage | 4 | 0.4 | 0.5 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 832x480x85f | LatentPreparationStage | 4 | 1.8 | 1.9 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 832x480x85f | TextEncodingStage | 4 | 434.9 | 435.6 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 832x480x85f | TimestepPreparationStage | 4 | 0.6 | 0.7 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 832x480x85f | decoding_stage | 4 | 6250.6 | 6253.0 |
