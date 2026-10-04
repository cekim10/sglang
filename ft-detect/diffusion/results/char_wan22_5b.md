Execution-quantum durations over 1 run(s). D_w_step = 2 x Q0.999(warm step); D_w_stage = 2 x max warm non-denoising stage (e.g. VAE decode); D_w = max of the two; D_global = max_w D_w.

| model | shape | pixels_frames | steps | requests | step_p50_ms | step_p99_ms | step_p999_ms | step_max_ms | gap_max_ms | warm_steps | warm_q_ms | warm_max_ms | max_first_after_launch_ms | max_first_of_shape_ms | max_after_idle_ms | D_w_step_s | D_w_all_s | stage_warm_max_ms | stage_warm_name | stage_cold_max_ms | D_w_stage_s | D_w_s | D_w_quantum | slack_x |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 832x480x17f | 6789120 | 54 | 3 | 167.3 | 2877.4 | 3494.5 | 3563.1 | 172.0 | 36 | 171.8 | 171.8 | 3563.1 |  |  | 0.344 | 7.126 | 1211.0 | decoding_stage | 7048.8 | 2.422 | 2.422 | decoding_stage | 4.9 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 832x480x49f | 19568640 | 54 | 3 | 418.4 | 429.6 | 429.7 | 429.7 | 429.9 | 36 | 429.7 | 429.7 |  | 429.1 |  | 0.859 | 0.86 | 3675.0 | decoding_stage |  | 7.35 | 7.35 | decoding_stage | 1.6 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 832x480x81f | 32348160 | 54 | 3 | 711.6 | 731.5 | 731.5 | 731.5 | 731.8 | 36 | 731.5 | 731.5 |  | 729.9 |  | 1.463 | 1.464 | 5970.8 | decoding_stage |  | 11.942 | 11.942 | decoding_stage | 1.0 |

**D_global = 11.94 s (set by 832x480x81f / decoding_stage); smallest D_w = 2.42 s; max slack = 5x.** Cold categories: `max_first_after_launch_ms`, `max_first_of_shape_ms`, `max_after_idle_ms` are the slowest step of such requests (a static watchdog must tolerate them; a progress-aware one can know they are coming).

Non-denoising stages (single execution quanta without step boundaries):

| model | shape | stage | n | p50_ms | max_ms |
|---|---|---|---|---|---|
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 1280x720x17f | InputValidationStage | 6 | 0.4 | 0.6 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 1280x720x17f | LatentPreparationStage | 6 | 1.8 | 1.9 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 1280x720x17f | TextEncodingStage | 6 | 429.8 | 434.3 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 1280x720x17f | TimestepPreparationStage | 6 | 0.5 | 0.6 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 1280x720x49f | InputValidationStage | 6 | 0.3 | 0.7 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 1280x720x49f | LatentPreparationStage | 6 | 1.7 | 1.9 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 1280x720x49f | TextEncodingStage | 6 | 428.3 | 431.0 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 1280x720x49f | TimestepPreparationStage | 6 | 0.4 | 0.6 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 1280x720x81f | InputValidationStage | 6 | 0.4 | 0.7 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 1280x720x81f | LatentPreparationStage | 6 | 1.8 | 1.9 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 1280x720x81f | TextEncodingStage | 6 | 429.8 | 435.8 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 1280x720x81f | TimestepPreparationStage | 6 | 0.5 | 0.6 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 832x480x17f | InputValidationStage | 6 | 0.4 | 0.6 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 832x480x17f | LatentPreparationStage | 6 | 1.8 | 2.9 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 832x480x17f | TextEncodingStage | 6 | 442.0 | 4302.4 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 832x480x17f | TimestepPreparationStage | 6 | 0.6 | 1.6 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 832x480x17f | decoding_stage | 6 | 1210.5 | 7048.8 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 832x480x49f | InputValidationStage | 6 | 0.4 | 0.6 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 832x480x49f | LatentPreparationStage | 6 | 1.8 | 2.0 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 832x480x49f | TextEncodingStage | 6 | 435.1 | 438.8 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 832x480x49f | TimestepPreparationStage | 6 | 0.6 | 0.7 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 832x480x49f | decoding_stage | 6 | 3629.4 | 3675.0 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 832x480x81f | InputValidationStage | 6 | 0.4 | 0.6 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 832x480x81f | LatentPreparationStage | 6 | 1.8 | 2.0 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 832x480x81f | TextEncodingStage | 6 | 433.8 | 434.4 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 832x480x81f | TimestepPreparationStage | 6 | 0.5 | 0.7 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 832x480x81f | decoding_stage | 6 | 5960.8 | 5970.8 |
