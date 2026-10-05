# Ideal warm retry vs in-place SP=2 -> SP=1 continuation (crossover audit)

Cells: T_retry / T_ours, both measured from the failure (detection included; >1 means ours finishes first). p* = earliest progress k/N at which ours wins. Steps: Wan2.2-TI2V-5B 50 (SGLang default sampling), Z-Image-Turbo 8 (cookbook). Test 3 itself ran 9 steps at 832x480x81, which nobody deploys.

## Calibration on Test 3 (832x480x81, 9 steps, failure at step 4)

| term | value |
|---|---|
| runs | 3 |
| step at SP=2 (s) | 0.72 |
| step at SP=1 (s) | 1.20 |
| SP=1 / SP=2 step | 1.66 |
| decode at SP=2 (s) | 5.96 |
| decode at SP=1 (s, from the next request) | 10.72 |
| SP=1 / SP=2 decode | 1.80 |
| text encode (s) | 0.43 |
| fixed request overhead (s) | 2.28 |
| detection (s) | 5.00 |
| switch (s) | 1.25 |
| first-SP=1-use residual in the failing request (s) | 4.22 |

Check: Test 3 measured failing request 32.8 s; B1 retry would have taken 20.1 s from the failure plus the 3.3 s already spent, so at this setting retry wins.

## Scenario: measured (SP=1 slowdown and first-use residual as measured in Test 3)

| workload | steps | full request (s) | p* | bound 1-q2/q1 | 20% | 40% | 50% | 60% | 80% | 95% | uniform: retry/ours | uniform: retry/best-of-both | source |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Wan2.2-TI2V-5B-Diffusers 512x512 | 50 | 8.8 | 0.90 | 0.40 | 0.66 | 0.74 | 0.78 | 0.82 | 0.94 | 1.05 | 0.77 | 1.00 | measured |
| Wan2.2-TI2V-5B-Diffusers 832x480x17f | 50 | 12.5 | 0.82 | 0.40 | 0.66 | 0.75 | 0.79 | 0.85 | 0.99 | 1.13 | 0.79 | 1.01 | measured |
| Wan2.2-TI2V-5B-Diffusers 832x480x81f | 50 | 45.2 | 0.56 | 0.39 | 0.70 | 0.85 | 0.94 | 1.06 | 1.42 | 1.94 | 0.93 | 1.13 | measured |
| Wan2.2-TI2V-5B-Diffusers 832x480x85f | 50 | 48.4 | 0.56 | 0.40 | 0.70 | 0.84 | 0.94 | 1.06 | 1.44 | 2.01 | 0.93 | 1.13 | measured |
| Wan2.2-TI2V-5B-Diffusers 1280x704x81f | 50 | 120.7 | 0.50 | 0.40 | 0.71 | 0.89 | 1.02 | 1.19 | 1.78 | 2.95 | 1.01 | 1.22 | measured |
| Z-Image-Turbo 256x256 | 8 | 2.9 | never | 0.40 | 0.60 | 0.60 | 0.61 | 0.61 | 0.61 | 0.62 | 0.60 | 1.00 | measured |
| Z-Image-Turbo 512x512 | 8 | 3.8 | never | 0.40 | 0.65 | 0.66 | 0.66 | 0.67 | 0.68 | 0.68 | 0.66 | 1.00 | measured |
| Z-Image-Turbo 1024x1024 | 8 | 4.4 | never | 0.25 | 0.64 | 0.66 | 0.67 | 0.69 | 0.70 | 0.72 | 0.67 | 1.00 | measured |
| Z-Image-Turbo 1536x1536 | 8 | 7.7 | never | 0.40 | 0.65 | 0.68 | 0.71 | 0.75 | 0.79 | 0.83 | 0.69 | 1.00 | measured |
| Wan2.2-TI2V-5B-Diffusers 832x480x49f | 50 | 27.8 | 0.64 | 0.40 | 0.69 | 0.81 | 0.88 | 0.98 | 1.24 | 1.59 | 0.88 | 1.08 | measured |
| Wan2.2-TI2V-5B-Diffusers 1024x1024 | 50 | 8.7 | 0.92 | 0.40 | 0.66 | 0.73 | 0.78 | 0.82 | 0.93 | 1.04 | 0.77 | 1.00 | measured |
| Wan2.2-TI2V-5B-Diffusers 1536x1536 | 50 | 13.4 | 0.72 | 0.40 | 0.68 | 0.77 | 0.83 | 0.90 | 1.08 | 1.28 | 0.83 | 1.03 | measured |
| Wan2.2-TI2V-5B-Diffusers 1280x704x17f | 50 | 24.9 | 0.64 | 0.40 | 0.69 | 0.81 | 0.88 | 0.98 | 1.24 | 1.58 | 0.87 | 1.08 | measured |
| Wan2.2-TI2V-5B-Diffusers 1280x704x49f | 50 | 64.5 | 0.54 | 0.40 | 0.70 | 0.86 | 0.97 | 1.11 | 1.56 | 2.30 | 0.96 | 1.16 | measured |
| Wan2.2-TI2V-5B-Diffusers 1280x704x121f | 50 | 202.3 | 0.48 | 0.40 | 0.72 | 0.91 | 1.05 | 1.24 | 1.94 | 3.54 | 1.04 | 1.25 | extrapolated (step ~ latent_frames^1.39, decode linear) |

## Scenario: pessimistic (SP=1 exactly 2x slower than SP=2 for steps and decode (perfect SP scaling))

| workload | steps | full request (s) | p* | bound 1-q2/q1 | 20% | 40% | 50% | 60% | 80% | 95% | uniform: retry/ours | uniform: retry/best-of-both | source |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Wan2.2-TI2V-5B-Diffusers 512x512 | 50 | 8.8 | 0.92 | 0.50 | 0.62 | 0.69 | 0.74 | 0.79 | 0.91 | 1.04 | 0.73 | 1.00 | measured |
| Wan2.2-TI2V-5B-Diffusers 832x480x17f | 50 | 12.5 | 0.88 | 0.50 | 0.61 | 0.69 | 0.74 | 0.80 | 0.94 | 1.11 | 0.73 | 1.01 | measured |
| Wan2.2-TI2V-5B-Diffusers 832x480x81f | 50 | 45.2 | 0.66 | 0.50 | 0.60 | 0.73 | 0.82 | 0.93 | 1.28 | 1.82 | 0.81 | 1.09 | measured |
| Wan2.2-TI2V-5B-Diffusers 832x480x85f | 50 | 48.4 | 0.66 | 0.50 | 0.60 | 0.74 | 0.83 | 0.94 | 1.30 | 1.88 | 0.82 | 1.10 | measured |
| Wan2.2-TI2V-5B-Diffusers 1280x704x81f | 50 | 120.7 | 0.60 | 0.50 | 0.61 | 0.76 | 0.88 | 1.03 | 1.57 | 2.70 | 0.86 | 1.16 | measured |
| Z-Image-Turbo 256x256 | 8 | 2.9 | never | 0.50 | 0.60 | 0.60 | 0.60 | 0.61 | 0.61 | 0.62 | 0.60 | 1.00 | measured |
| Z-Image-Turbo 512x512 | 8 | 3.8 | never | 0.50 | 0.65 | 0.65 | 0.66 | 0.67 | 0.67 | 0.68 | 0.66 | 1.00 | measured |
| Z-Image-Turbo 1024x1024 | 8 | 4.4 | never | 0.50 | 0.61 | 0.63 | 0.64 | 0.66 | 0.69 | 0.71 | 0.63 | 1.00 | measured |
| Z-Image-Turbo 1536x1536 | 8 | 7.7 | never | 0.50 | 0.61 | 0.64 | 0.68 | 0.72 | 0.76 | 0.81 | 0.66 | 1.00 | measured |
| Wan2.2-TI2V-5B-Diffusers 832x480x49f | 50 | 27.8 | 0.72 | 0.50 | 0.60 | 0.72 | 0.79 | 0.88 | 1.15 | 1.51 | 0.78 | 1.06 | measured |
| Wan2.2-TI2V-5B-Diffusers 1024x1024 | 50 | 8.7 | 0.94 | 0.50 | 0.62 | 0.69 | 0.73 | 0.78 | 0.91 | 1.04 | 0.73 | 1.00 | measured |
| Wan2.2-TI2V-5B-Diffusers 1536x1536 | 50 | 13.4 | 0.78 | 0.50 | 0.61 | 0.71 | 0.77 | 0.84 | 1.03 | 1.26 | 0.76 | 1.03 | measured |
| Wan2.2-TI2V-5B-Diffusers 1280x704x17f | 50 | 24.9 | 0.70 | 0.50 | 0.61 | 0.72 | 0.79 | 0.89 | 1.15 | 1.51 | 0.79 | 1.06 | measured |
| Wan2.2-TI2V-5B-Diffusers 1280x704x49f | 50 | 64.5 | 0.62 | 0.50 | 0.61 | 0.75 | 0.85 | 0.97 | 1.40 | 2.14 | 0.83 | 1.12 | measured |
| Wan2.2-TI2V-5B-Diffusers 1280x704x121f | 50 | 202.3 | 0.58 | 0.50 | 0.61 | 0.78 | 0.90 | 1.06 | 1.69 | 3.20 | 0.88 | 1.19 | extrapolated (step ~ latent_frames^1.39, decode linear) |

## Scenario: prewarmed (as measured, but the first-SP=1-use residual removed (SP=1 shapes warmed in advance))

| workload | steps | full request (s) | p* | bound 1-q2/q1 | 20% | 40% | 50% | 60% | 80% | 95% | uniform: retry/ours | uniform: retry/best-of-both | source |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Wan2.2-TI2V-5B-Diffusers 512x512 | 50 | 8.8 | 0.48 | 0.40 | 0.83 | 0.95 | 1.02 | 1.10 | 1.31 | 1.54 | 1.01 | 1.11 | measured |
| Wan2.2-TI2V-5B-Diffusers 832x480x17f | 50 | 12.5 | 0.54 | 0.40 | 0.79 | 0.91 | 0.98 | 1.07 | 1.29 | 1.55 | 0.97 | 1.10 | measured |
| Wan2.2-TI2V-5B-Diffusers 832x480x81f | 50 | 45.2 | 0.50 | 0.39 | 0.75 | 0.91 | 1.02 | 1.16 | 1.61 | 2.32 | 1.01 | 1.18 | measured |
| Wan2.2-TI2V-5B-Diffusers 832x480x85f | 50 | 48.4 | 0.50 | 0.40 | 0.74 | 0.90 | 1.02 | 1.16 | 1.62 | 2.39 | 1.00 | 1.18 | measured |
| Wan2.2-TI2V-5B-Diffusers 1280x704x81f | 50 | 120.7 | 0.48 | 0.40 | 0.73 | 0.92 | 1.06 | 1.24 | 1.89 | 3.27 | 1.04 | 1.24 | measured |
| Z-Image-Turbo 256x256 | 8 | 2.9 | never | 0.40 | 0.88 | 0.89 | 0.90 | 0.90 | 0.91 | 0.92 | 0.89 | 1.00 | measured |
| Z-Image-Turbo 512x512 | 8 | 3.8 | 0.88 | 0.40 | 0.95 | 0.96 | 0.97 | 0.98 | 1.00 | 1.01 | 0.97 | 1.00 | measured |
| Z-Image-Turbo 1024x1024 | 8 | 4.4 | 0.75 | 0.25 | 0.91 | 0.93 | 0.96 | 0.99 | 1.02 | 1.06 | 0.95 | 1.01 | measured |
| Z-Image-Turbo 1536x1536 | 8 | 7.7 | 0.75 | 0.40 | 0.82 | 0.87 | 0.93 | 0.99 | 1.07 | 1.15 | 0.90 | 1.02 | measured |
| Wan2.2-TI2V-5B-Diffusers 832x480x49f | 50 | 27.8 | 0.52 | 0.40 | 0.75 | 0.90 | 1.00 | 1.12 | 1.48 | 1.99 | 0.99 | 1.15 | measured |
| Wan2.2-TI2V-5B-Diffusers 1024x1024 | 50 | 8.7 | 0.48 | 0.40 | 0.83 | 0.95 | 1.02 | 1.10 | 1.30 | 1.53 | 1.01 | 1.10 | measured |
| Wan2.2-TI2V-5B-Diffusers 1536x1536 | 50 | 13.4 | 0.48 | 0.40 | 0.80 | 0.94 | 1.03 | 1.14 | 1.44 | 1.82 | 1.02 | 1.14 | measured |
| Wan2.2-TI2V-5B-Diffusers 1280x704x17f | 50 | 24.9 | 0.50 | 0.40 | 0.76 | 0.91 | 1.01 | 1.13 | 1.50 | 2.03 | 1.00 | 1.16 | measured |
| Wan2.2-TI2V-5B-Diffusers 1280x704x49f | 50 | 64.5 | 0.48 | 0.40 | 0.74 | 0.91 | 1.03 | 1.19 | 1.72 | 2.67 | 1.02 | 1.21 | measured |
| Wan2.2-TI2V-5B-Diffusers 1280x704x121f | 50 | 202.3 | 0.46 | 0.40 | 0.73 | 0.93 | 1.08 | 1.27 | 2.02 | 3.81 | 1.06 | 1.27 | extrapolated (step ~ latent_frames^1.39, decode linear) |

## Gate (pre-agreed): measured scenario, video workloads at recommended steps

- p* <= 0.5 and >= 1.5x at 80% progress: 1280x704x81f, 1280x704x121f
- crossover only after 50%: 832x480x17f (p*=0.82), 832x480x81f (p*=0.56), 832x480x85f (p*=0.56), 832x480x49f (p*=0.64), 1280x704x17f (p*=0.64), 1280x704x49f (p*=0.54)
- ours never wins: none
