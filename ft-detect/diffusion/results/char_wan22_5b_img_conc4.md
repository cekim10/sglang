Execution-quantum durations over 1 run(s). D_w_step = 2 x Q0.999(warm step); D_w_stage = 2 x max warm non-denoising stage (e.g. VAE decode); D_w = max of the two; D_global = max_w D_w.

| model | shape | pixels_frames | steps | requests | step_p50_ms | step_p99_ms | step_p999_ms | step_max_ms | gap_max_ms | warm_steps | warm_q_ms | warm_max_ms | max_first_after_launch_ms | max_first_of_shape_ms | max_after_idle_ms | D_w_step_s | D_w_all_s | stage_warm_max_ms | stage_warm_name | stage_cold_max_ms | D_w_stage_s | D_w_s | D_w_quantum | slack_x |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 512x512 | 262144 | 468 | 26 | 83.8 | 119.5 | 2605.1 | 2988.8 | 90.6 | 450 | 120.3 | 121.0 | 2988.8 |  |  | 0.241 | 5.978 | 565.3 | TextEncodingStage | 2333.1 | 1.131 | 1.131 | TextEncodingStage | 22.6 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 1024x1024 | 1048576 | 522 | 29 | 92.9 | 118.2 | 118.8 | 118.9 | 96.4 | 504 | 118.5 | 118.6 |  | 118.9 |  | 0.237 | 0.238 | 567.2 | TextEncodingStage |  | 1.134 | 1.134 | TextEncodingStage | 22.6 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 1536x1536 | 2359296 | 630 | 35 | 195.4 | 202.0 | 202.4 | 202.5 | 202.6 | 612 | 202.4 | 202.5 |  | 197.3 |  | 0.405 | 0.405 | 591.1 | decoding_stage |  | 1.182 | 1.182 | decoding_stage | 21.7 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 1280x704x17f | 15319040 | 54 | 3 | 378.1 | 388.2 | 388.4 | 388.4 | 388.6 | 36 | 388.4 | 388.4 |  | 385.7 |  | 0.777 | 0.777 | 2828.9 | decoding_stage |  | 5.658 | 5.658 | decoding_stage | 4.5 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 1280x704x49f | 44154880 | 54 | 3 | 1050.8 | 1081.7 | 1081.7 | 1081.7 | 1082.0 | 36 | 1081.7 | 1081.7 |  | 1072.8 |  | 2.163 | 2.164 | 7758.9 | decoding_stage |  | 15.518 | 15.518 | decoding_stage | 1.6 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 1280x704x81f | 72990720 | 54 | 3 | 2047.5 | 2105.3 | 2105.5 | 2105.5 | 2105.7 | 36 | 2105.5 | 2105.5 |  | 2104.7 |  | 4.211 | 4.211 | 12797.5 | decoding_stage |  | 25.595 | 25.595 | decoding_stage | 1.0 |

**D_global = 25.59 s (set by 1280x704x81f / decoding_stage); smallest D_w = 1.13 s; max slack = 23x.** Cold categories: `max_first_after_launch_ms`, `max_first_of_shape_ms`, `max_after_idle_ms` are the slowest step of such requests (a static watchdog must tolerate them; a progress-aware one can know they are coming).

Non-denoising stages (single execution quanta without step boundaries):

| model | shape | stage | n | p50_ms | max_ms |
|---|---|---|---|---|---|
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 1024x1024 | InputValidationStage | 58 | 0.2 | 0.6 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 1024x1024 | LatentPreparationStage | 58 | 1.8 | 2.2 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 1024x1024 | TextEncodingStage | 58 | 511.1 | 567.2 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 1024x1024 | TimestepPreparationStage | 58 | 0.5 | 0.7 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 1024x1024 | decoding_stage | 58 | 29.8 | 49.6 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 1280x704x17f | InputValidationStage | 6 | 0.5 | 0.9 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 1280x704x17f | LatentPreparationStage | 6 | 1.9 | 2.2 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 1280x704x17f | TextEncodingStage | 6 | 434.4 | 442.0 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 1280x704x17f | TimestepPreparationStage | 6 | 0.6 | 0.7 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 1280x704x17f | decoding_stage | 6 | 2723.7 | 2828.9 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 1280x704x49f | InputValidationStage | 6 | 0.5 | 1.7 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 1280x704x49f | LatentPreparationStage | 6 | 1.8 | 2.0 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 1280x704x49f | TextEncodingStage | 6 | 442.7 | 518.8 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 1280x704x49f | TimestepPreparationStage | 6 | 0.6 | 0.7 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 1280x704x49f | decoding_stage | 6 | 7731.7 | 7758.9 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 1280x704x81f | InputValidationStage | 6 | 0.4 | 0.6 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 1280x704x81f | LatentPreparationStage | 6 | 1.8 | 2.0 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 1280x704x81f | TextEncodingStage | 6 | 440.9 | 441.6 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 1280x704x81f | TimestepPreparationStage | 6 | 0.6 | 0.7 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 1280x704x81f | decoding_stage | 6 | 12733.3 | 12797.5 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 1536x1536 | InputValidationStage | 70 | 0.2 | 0.6 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 1536x1536 | LatentPreparationStage | 70 | 1.8 | 2.4 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 1536x1536 | TextEncodingStage | 70 | 510.2 | 567.6 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 1536x1536 | TimestepPreparationStage | 70 | 0.5 | 0.7 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 1536x1536 | decoding_stage | 70 | 531.4 | 591.1 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 512x512 | InputValidationStage | 52 | 0.2 | 0.8 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 512x512 | LatentPreparationStage | 52 | 1.8 | 2.2 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 512x512 | TextEncodingStage | 52 | 509.6 | 2333.1 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 512x512 | TimestepPreparationStage | 52 | 0.5 | 0.8 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 512x512 | decoding_stage | 52 | 15.0 | 355.6 |
