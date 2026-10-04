Merged over 3 characterization CSV(s) (char_zimage.csv, char_wan22_5b.csv, char_wan22_5b_img_conc4.csv), 2 model(s), 13 workload(s).

| model | shape | pixels_frames | steps | requests | step_p50_ms | step_p999_ms | step_max_ms | warm_q_ms | max_first_after_launch_ms | max_first_of_shape_ms | max_after_idle_ms | stage_warm_max_ms | stage_cold_max_ms | D_w_step_s | D_w_stage_s | D_w_quantum | D_w_s | D_cold_s | slack_x | slack_incl_cold_x |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Tongyi-MAI/Z-Image-Turbo | 256x256 | 65536 | 108 | 6 | 42.5 | 267.5 | 274.5 | 45.5 | 274.5 |  | 47.7 | 273.8 | 1713.3 | 0.091 | 0.548 | TextEncodingStage | 0.548 | 3.43 | 46.7 | 46.7 |
| Tongyi-MAI/Z-Image-Turbo | 512x512 | 262144 | 108 | 6 | 65.0 | 72.7 | 72.8 | 72.5 |  | 72.4 |  | 267.4 |  | 0.145 | 0.535 | TextEncodingStage | 0.535 | 0.14 | 47.8 | 47.8 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 512x512 | 262144 | 468 | 26 | 83.8 | 2605.1 | 2988.8 | 120.3 | 2988.8 |  |  | 565.3 | 2333.1 | 0.241 | 1.131 | TextEncodingStage | 1.131 | 5.98 | 22.6 | 22.6 |
| Tongyi-MAI/Z-Image-Turbo | 1024x1024 | 1048576 | 108 | 6 | 218.9 | 221.3 | 221.3 | 221.3 |  | 220.4 |  | 323.3 |  | 0.443 | 0.647 | TextEncodingStage | 0.647 | 0.44 | 39.6 | 39.6 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 1024x1024 | 1048576 | 522 | 29 | 92.9 | 118.8 | 118.9 | 118.5 |  | 118.9 |  | 567.2 |  | 0.237 | 1.134 | TextEncodingStage | 1.134 | 0.24 | 22.6 | 22.6 |
| Tongyi-MAI/Z-Image-Turbo | 1536x1536 | 2359296 | 108 | 6 | 529.2 | 532.4 | 532.4 | 532.4 |  | 527.8 |  | 913.3 |  | 1.065 | 1.827 | decoding_stage | 1.827 | 1.06 | 14.0 | 14.0 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 1536x1536 | 2359296 | 630 | 35 | 195.4 | 202.4 | 202.5 | 202.4 |  | 197.3 |  | 591.1 |  | 0.405 | 1.182 | decoding_stage | 1.182 | 0.39 | 21.7 | 21.7 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 832x480x17f | 6789120 | 54 | 3 | 167.3 | 3494.5 | 3563.1 | 171.8 | 3563.1 |  |  | 1211.0 | 7048.8 | 0.344 | 2.422 | decoding_stage | 2.422 | 14.1 | 10.6 | 10.6 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 1280x704x17f | 15319040 | 54 | 3 | 378.1 | 388.4 | 388.4 | 388.4 |  | 385.7 |  | 2828.9 |  | 0.777 | 5.658 | decoding_stage | 5.658 | 0.77 | 4.5 | 4.5 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 832x480x49f | 19568640 | 54 | 3 | 418.4 | 429.7 | 429.7 | 429.7 |  | 429.1 |  | 3675.0 |  | 0.859 | 7.35 | decoding_stage | 7.35 | 0.86 | 3.5 | 3.5 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 832x480x81f | 32348160 | 54 | 3 | 711.6 | 731.5 | 731.5 | 731.5 |  | 729.9 |  | 5970.8 |  | 1.463 | 11.942 | decoding_stage | 11.942 | 1.46 | 2.1 | 2.1 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 1280x704x49f | 44154880 | 54 | 3 | 1050.8 | 1081.7 | 1081.7 | 1081.7 |  | 1072.8 |  | 7758.9 |  | 2.163 | 15.518 | decoding_stage | 15.518 | 2.15 | 1.6 | 1.6 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | 1280x704x81f | 72990720 | 54 | 3 | 2047.5 | 2105.5 | 2105.5 | 2105.5 |  | 2104.7 |  | 12797.5 |  | 4.211 | 25.595 | decoding_stage | 25.595 | 4.21 | 1.0 | 1.0 |

**D_global (warm) = 25.59 s; D_global including cold steps = 25.59 s; smallest D_w = 0.535 s; max slack = 48x (incl. cold: 48x).**

```
safe deadline per workload = largest warm execution quantum x2 (log axis 0.01 s .. 30.7 s); '#' = D_w, 'c' = worst cold quantum, '|' = D_global=25.59 s

Z-Image-Turbo               256x256 ########################          c           |  D_w=0.55s (TextEncodingStage) slack=47x
Z-Image-Turbo               512x512 ###############c########                      |  D_w=0.54s (TextEncodingStage) slack=48x
Wan2.2-TI2V-5B-Dif          512x512 #############################        c        |  D_w=1.13s (TextEncodingStage) slack=23x
Z-Image-Turbo             1024x1024 ######################c##                     |  D_w=0.65s (TextEncodingStage) slack=40x
Wan2.2-TI2V-5B-Dif        1024x1024 ###################c#########                 |  D_w=1.13s (TextEncodingStage) slack=23x
Z-Image-Turbo             1536x1536 ###########################c###               |  D_w=1.83s (decoding_stage) slack=14x
Wan2.2-TI2V-5B-Dif        1536x1536 #####################c#######                 |  D_w=1.18s (decoding_stage) slack=22x
Wan2.2-TI2V-5B-Dif      832x480x17f #################################         c   |  D_w=2.42s (decoding_stage) slack=11x
Wan2.2-TI2V-5B-Dif     1280x704x17f #########################c############        |  D_w=5.66s (decoding_stage) slack=4x
Wan2.2-TI2V-5B-Dif      832x480x49f ##########################c#############      |  D_w=7.35s (decoding_stage) slack=4x
Wan2.2-TI2V-5B-Dif      832x480x81f #############################c############    |  D_w=11.94s (decoding_stage) slack=2x
Wan2.2-TI2V-5B-Dif     1280x704x49f ###############################c############  |  D_w=15.52s (decoding_stage) slack=2x
Wan2.2-TI2V-5B-Dif     1280x704x81f ###################################c##########|  D_w=25.59s (decoding_stage) slack=1x
```
