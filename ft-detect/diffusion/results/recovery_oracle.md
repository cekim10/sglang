# Recovery oracle (cost model from measured inputs) -- DRAFT: q_sp1 measured for 1024^2 and 480p x 81 only, other shapes extrapolated by model ratio; decode_sp1 assumed = decode_sp2; no A14B, SP degrees 1-2 only

alpha=2.0, fence in [0.1, 8.0] s, restore=0.02 s, N in [9, 20, 30, 50], failure at k/N in [0.1, 0.25, 0.5, 0.75, 0.9], standby in ['none', 'sp1', 'sp1+sp2']. 1680 cells.

## Inputs per workload

| model                    | shape        |    pixels_frames |   q1_s |   q2_s | q1_extrapolated   |   decode1_s |   decode2_s |   encode_s |   cold_ready_s |   state_MiB |
|:-------------------------|:-------------|-----------------:|-------:|-------:|:------------------|------------:|------------:|-----------:|---------------:|------------:|
| Wan2.2-TI2V-5B-Diffusers | 512x512      | 262144           |  0.206 |  0.12  | True              |       0.015 |       0.015 |      0.51  |           37   |       0.751 |
| Wan2.2-TI2V-5B-Diffusers | 1024x1024    |      1.04858e+06 |  0.203 |  0.118 | True              |       0.03  |       0.03  |      0.511 |           37   |     nan     |
| Wan2.2-TI2V-5B-Diffusers | 1536x1536    |      2.3593e+06  |  0.346 |  0.202 | True              |       0.531 |       0.531 |      0.51  |           37   |     nan     |
| Wan2.2-TI2V-5B-Diffusers | 832x480x17f  |      6.78912e+06 |  0.294 |  0.172 | True              |       1.21  |       1.21  |      0.442 |           37   |       5.713 |
| Wan2.2-TI2V-5B-Diffusers | 1280x704x17f |      1.5319e+07  |  0.665 |  0.388 | True              |       2.724 |       2.724 |      0.434 |           37   |     nan     |
| Wan2.2-TI2V-5B-Diffusers | 832x480x49f  |      1.95686e+07 |  0.736 |  0.43  | True              |       3.629 |       3.629 |      0.435 |           37   |     nan     |
| Wan2.2-TI2V-5B-Diffusers | 832x480x81f  |      3.23482e+07 |  1.198 |  0.732 | False             |       5.961 |       5.961 |      0.434 |           37   |      23.995 |
| Wan2.2-TI2V-5B-Diffusers | 832x480x85f  |      3.39456e+07 |  1.351 |  0.789 | True              |       6.251 |       6.251 |      0.435 |           37   |      25.137 |
| Wan2.2-TI2V-5B-Diffusers | 1280x704x49f |      4.41549e+07 |  1.852 |  1.082 | True              |       7.732 |       7.732 |      0.443 |           37   |     nan     |
| Wan2.2-TI2V-5B-Diffusers | 1280x704x81f |      7.29907e+07 |  3.604 |  2.106 | True              |      12.733 |      12.733 |      0.441 |           37   |      54.14  |
| Z-Image-Turbo            | 256x256      |  65536           |  0.061 |  0.046 | True              |       0.007 |       0.007 |      0.267 |           24.6 |     nan     |
| Z-Image-Turbo            | 512x512      | 262144           |  0.089 |  0.067 | True              |       0.068 |       0.068 |      0.957 |           24.6 |       0.062 |
| Z-Image-Turbo            | 1024x1024    |      1.04858e+06 |  0.295 |  0.221 | False             |       0.054 |       0.054 |      0.32  |           24.6 |     nan     |
| Z-Image-Turbo            | 1536x1536    |      2.3593e+06  |  0.703 |  0.528 | True              |       0.903 |       0.903 |      0.288 |           24.6 |       0.562 |

## Headline

- cells where the best action beats the runner-up by >= 2x: **27.2%** (median G 1.15)
- best action distribution: {'restore_sp2': 1128, 'degrade': 552}
- static policy **always degrade**: median regret 3%, p95 46%, max 67%, cells with regret >= 100%: 0%, cells with regret >= 10%: 33%
- static policy **always restore_sp2**: median regret 0%, p95 971%, max 8824%, cells with regret >= 100%: 27%, cells with regret >= 10%: 32%
- static policy **always restart**: median regret 40%, p95 1060%, max 9044%, cells with regret >= 100%: 37%, cells with regret >= 10%: 74%
- static policy **two-rule: restore_sp2 if both spares are warm, else degrade**: median regret 0%, p95 14%, max 45%, cells with regret >= 100%: 0%, cells with regret >= 10%: 8%
- static policy **crossover rule: restore_sp2 iff remaining x (q1-q2) > extra ready time**: median regret 0%, p95 0%, max 0%, cells with regret >= 100%: 0%, cells with regret >= 10%: 0%
- cells where the two-rule policy loses >= 10%: 130 of 1680; standby={'none': 126, 'sp1': 4}, remaining steps median 24, models {'Wan2.2-TI2V-5B-Diffusers': 123, 'Z-Image-Turbo': 7}

## GO/KILL reading (criteria fixed in advance)

- KILL the planner if one simple static policy has p95 regret <= 10% and no large worst case: two-rule policy p95 = 14%, max = 45%.
- GO needs >= 2x penalties for every strong static policy in a realistic region: fraction of cells with G >= 2 is 27%, but those cells are {'sp1': 457} by standby and are all resolved by the two-rule policy (its max regret there: 0%).

## Regime maps (best action; fence = 0.1 s; cell = N x remaining)

### Wan2.2-TI2V-5B-Diffusers 832x480x81f (q1=1.198 s, q2=0.732 s)

standby = none:

| N \\ remaining | 1 | 2 | 3 | 5 | 7 | 8 | 10 | 12 | 15 | 18 | 22 | 25 | 27 | 38 | 45 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 9 | restore_sp2 (G=1.0) | restore_sp2 (G=1.0) |  | restore_sp2 (G=1.0) | restore_sp2 (G=1.0) | restore_sp2 (G=1.0) |  |  |  |  |  |  |  |  |  |
| 20 |  | restore_sp2 (G=1.0) |  | restore_sp2 (G=1.0) |  |  | restore_sp2 (G=1.1) |  | restore_sp2 (G=1.1) | restore_sp2 (G=1.0) |  |  |  |  |  |
| 30 |  |  | restore_sp2 (G=1.0) |  |  | restore_sp2 (G=1.1) |  |  | restore_sp2 (G=1.1) |  | restore_sp2 (G=1.1) |  | restore_sp2 (G=1.0) |  |  |
| 50 |  |  |  | restore_sp2 (G=1.0) |  |  |  | restore_sp2 (G=1.1) |  |  |  | restore_sp2 (G=1.2) |  | restore_sp2 (G=1.1) | restore_sp2 (G=1.1) |

standby = sp1:

| N \\ remaining | 1 | 2 | 3 | 5 | 7 | 8 | 10 | 12 | 15 | 18 | 22 | 25 | 27 | 38 | 45 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 9 | degrade (G=5.2) | degrade (G=4.6) |  | degrade (G=3.6) | degrade (G=3.1) | degrade (G=2.9) |  |  |  |  |  |  |  |  |  |
| 20 |  | degrade (G=4.6) |  | degrade (G=3.6) |  |  | degrade (G=2.7) |  | degrade (G=2.2) | degrade (G=2.0) |  |  |  |  |  |
| 30 |  |  | degrade (G=4.2) |  |  | degrade (G=2.9) |  |  | degrade (G=2.2) |  | degrade (G=1.8) |  | degrade (G=1.6) |  |  |
| 50 |  |  |  | degrade (G=3.6) |  |  |  | degrade (G=2.4) |  |  |  | degrade (G=1.7) |  | degrade (G=1.4) | degrade (G=1.3) |

standby = sp1+sp2:

| N \\ remaining | 1 | 2 | 3 | 5 | 7 | 8 | 10 | 12 | 15 | 18 | 22 | 25 | 27 | 38 | 45 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 9 | restore_sp2 (G=1.1) | restore_sp2 (G=1.1) |  | restore_sp2 (G=1.2) | restore_sp2 (G=1.1) | restore_sp2 (G=1.1) |  |  |  |  |  |  |  |  |  |
| 20 |  | restore_sp2 (G=1.1) |  | restore_sp2 (G=1.2) |  |  | restore_sp2 (G=1.3) |  | restore_sp2 (G=1.2) | restore_sp2 (G=1.1) |  |  |  |  |  |
| 30 |  |  | restore_sp2 (G=1.1) |  |  | restore_sp2 (G=1.3) |  |  | restore_sp2 (G=1.4) |  | restore_sp2 (G=1.3) |  | restore_sp2 (G=1.1) |  |  |
| 50 |  |  |  | restore_sp2 (G=1.2) |  |  |  | restore_sp2 (G=1.3) |  |  |  | restore_sp2 (G=1.5) |  | restore_sp2 (G=1.3) | restore_sp2 (G=1.1) |

### Wan2.2-TI2V-5B-Diffusers 1280x704x81f (q1=3.604 s, q2=2.106 s, q1 extrapolated)

standby = none:

| N \\ remaining | 1 | 2 | 3 | 5 | 7 | 8 | 10 | 12 | 15 | 18 | 22 | 25 | 27 | 38 | 45 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 9 | restore_sp2 (G=1.0) | restore_sp2 (G=1.1) |  | restore_sp2 (G=1.1) | restore_sp2 (G=1.1) | restore_sp2 (G=1.0) |  |  |  |  |  |  |  |  |  |
| 20 |  | restore_sp2 (G=1.1) |  | restore_sp2 (G=1.1) |  |  | restore_sp2 (G=1.2) |  | restore_sp2 (G=1.1) | restore_sp2 (G=1.1) |  |  |  |  |  |
| 30 |  |  | restore_sp2 (G=1.1) |  |  | restore_sp2 (G=1.2) |  |  | restore_sp2 (G=1.3) |  | restore_sp2 (G=1.2) |  | restore_sp2 (G=1.1) |  |  |
| 50 |  |  |  | restore_sp2 (G=1.1) |  |  |  | restore_sp2 (G=1.2) |  |  |  | restore_sp2 (G=1.4) |  | restore_sp2 (G=1.2) | restore_sp2 (G=1.1) |

standby = sp1:

| N \\ remaining | 1 | 2 | 3 | 5 | 7 | 8 | 10 | 12 | 15 | 18 | 22 | 25 | 27 | 38 | 45 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 9 | degrade (G=2.7) | degrade (G=2.4) |  | degrade (G=1.8) | degrade (G=1.6) | degrade (G=1.5) |  |  |  |  |  |  |  |  |  |
| 20 |  | degrade (G=2.4) |  | degrade (G=1.8) |  |  | degrade (G=1.4) |  | degrade (G=1.2) | degrade (G=1.1) |  |  |  |  |  |
| 30 |  |  | degrade (G=2.2) |  |  | degrade (G=1.5) |  |  | degrade (G=1.2) |  | degrade (G=1.0) |  | restore_sp2 (G=1.0) |  |  |
| 50 |  |  |  | degrade (G=1.8) |  |  |  | degrade (G=1.3) |  |  |  | restore_sp2 (G=1.0) |  | restore_sp2 (G=1.1) | restore_sp2 (G=1.1) |

standby = sp1+sp2:

| N \\ remaining | 1 | 2 | 3 | 5 | 7 | 8 | 10 | 12 | 15 | 18 | 22 | 25 | 27 | 38 | 45 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 9 | restore_sp2 (G=1.1) | restore_sp2 (G=1.1) |  | restore_sp2 (G=1.3) | restore_sp2 (G=1.1) | restore_sp2 (G=1.1) |  |  |  |  |  |  |  |  |  |
| 20 |  | restore_sp2 (G=1.1) |  | restore_sp2 (G=1.3) |  |  | restore_sp2 (G=1.4) |  | restore_sp2 (G=1.2) | restore_sp2 (G=1.1) |  |  |  |  |  |
| 30 |  |  | restore_sp2 (G=1.2) |  |  | restore_sp2 (G=1.4) |  |  | restore_sp2 (G=1.5) |  | restore_sp2 (G=1.3) |  | restore_sp2 (G=1.1) |  |  |
| 50 |  |  |  | restore_sp2 (G=1.3) |  |  |  | restore_sp2 (G=1.4) |  |  |  | restore_sp2 (G=1.5) |  | restore_sp2 (G=1.3) | restore_sp2 (G=1.1) |

### Z-Image-Turbo 1024x1024 (q1=0.295 s, q2=0.221 s)

standby = none:

| N \\ remaining | 1 | 2 | 3 | 5 | 7 | 8 | 10 | 12 | 15 | 18 | 22 | 25 | 27 | 38 | 45 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 9 | restore_sp2 (G=1.0) | restore_sp2 (G=1.0) |  | restore_sp2 (G=1.0) | restore_sp2 (G=1.0) | restore_sp2 (G=1.0) |  |  |  |  |  |  |  |  |  |
| 20 |  | restore_sp2 (G=1.0) |  | restore_sp2 (G=1.0) |  |  | restore_sp2 (G=1.0) |  | restore_sp2 (G=1.0) | restore_sp2 (G=1.0) |  |  |  |  |  |
| 30 |  |  | restore_sp2 (G=1.0) |  |  | restore_sp2 (G=1.0) |  |  | restore_sp2 (G=1.0) |  | restore_sp2 (G=1.1) |  | restore_sp2 (G=1.0) |  |  |
| 50 |  |  |  | restore_sp2 (G=1.0) |  |  |  | restore_sp2 (G=1.0) |  |  |  | restore_sp2 (G=1.1) |  | restore_sp2 (G=1.1) | restore_sp2 (G=1.0) |

standby = sp1:

| N \\ remaining | 1 | 2 | 3 | 5 | 7 | 8 | 10 | 12 | 15 | 18 | 22 | 25 | 27 | 38 | 45 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 9 | degrade (G=27.9) | degrade (G=21.3) |  | degrade (G=12.6) | degrade (G=10.0) | degrade (G=9.1) |  |  |  |  |  |  |  |  |  |
| 20 |  | degrade (G=21.3) |  | degrade (G=12.6) |  |  | degrade (G=7.7) |  | degrade (G=5.7) | degrade (G=4.9) |  |  |  |  |  |
| 30 |  |  | degrade (G=17.2) |  |  | degrade (G=9.1) |  |  | degrade (G=5.7) |  | degrade (G=4.2) |  | degrade (G=3.6) |  |  |
| 50 |  |  |  | degrade (G=12.6) |  |  |  | degrade (G=6.7) |  |  |  | degrade (G=3.8) |  | degrade (G=2.8) | degrade (G=2.5) |

standby = sp1+sp2:

| N \\ remaining | 1 | 2 | 3 | 5 | 7 | 8 | 10 | 12 | 15 | 18 | 22 | 25 | 27 | 38 | 45 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 9 | restore_sp2 (G=1.1) | restore_sp2 (G=1.1) |  | restore_sp2 (G=1.2) | restore_sp2 (G=1.2) | restore_sp2 (G=1.2) |  |  |  |  |  |  |  |  |  |
| 20 |  | restore_sp2 (G=1.1) |  | restore_sp2 (G=1.2) |  |  | restore_sp2 (G=1.3) |  | restore_sp2 (G=1.3) | restore_sp2 (G=1.2) |  |  |  |  |  |
| 30 |  |  | restore_sp2 (G=1.2) |  |  | restore_sp2 (G=1.2) |  |  | restore_sp2 (G=1.3) |  | restore_sp2 (G=1.3) |  | restore_sp2 (G=1.1) |  |  |
| 50 |  |  |  | restore_sp2 (G=1.2) |  |  |  | restore_sp2 (G=1.3) |  |  |  | restore_sp2 (G=1.3) |  | restore_sp2 (G=1.3) | restore_sp2 (G=1.1) |

### Z-Image-Turbo 1536x1536 (q1=0.703 s, q2=0.528 s, q1 extrapolated)

standby = none:

| N \\ remaining | 1 | 2 | 3 | 5 | 7 | 8 | 10 | 12 | 15 | 18 | 22 | 25 | 27 | 38 | 45 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 9 | restore_sp2 (G=1.0) | restore_sp2 (G=1.0) |  | restore_sp2 (G=1.0) | restore_sp2 (G=1.0) | restore_sp2 (G=1.0) |  |  |  |  |  |  |  |  |  |
| 20 |  | restore_sp2 (G=1.0) |  | restore_sp2 (G=1.0) |  |  | restore_sp2 (G=1.1) |  | restore_sp2 (G=1.1) | restore_sp2 (G=1.0) |  |  |  |  |  |
| 30 |  |  | restore_sp2 (G=1.0) |  |  | restore_sp2 (G=1.0) |  |  | restore_sp2 (G=1.1) |  | restore_sp2 (G=1.1) |  | restore_sp2 (G=1.0) |  |  |
| 50 |  |  |  | restore_sp2 (G=1.0) |  |  |  | restore_sp2 (G=1.1) |  |  |  | restore_sp2 (G=1.1) |  | restore_sp2 (G=1.1) | restore_sp2 (G=1.1) |

standby = sp1:

| N \\ remaining | 1 | 2 | 3 | 5 | 7 | 8 | 10 | 12 | 15 | 18 | 22 | 25 | 27 | 38 | 45 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 9 | degrade (G=9.8) | degrade (G=8.0) |  | degrade (G=5.2) | degrade (G=4.3) | degrade (G=4.0) |  |  |  |  |  |  |  |  |  |
| 20 |  | degrade (G=8.0) |  | degrade (G=5.2) |  |  | degrade (G=3.5) |  | degrade (G=2.7) | degrade (G=2.5) |  |  |  |  |  |
| 30 |  |  | degrade (G=6.8) |  |  | degrade (G=4.0) |  |  | degrade (G=2.7) |  | degrade (G=2.2) |  | degrade (G=1.9) |  |  |
| 50 |  |  |  | degrade (G=5.2) |  |  |  | degrade (G=3.1) |  |  |  | degrade (G=2.0) |  | degrade (G=1.6) | degrade (G=1.5) |

standby = sp1+sp2:

| N \\ remaining | 1 | 2 | 3 | 5 | 7 | 8 | 10 | 12 | 15 | 18 | 22 | 25 | 27 | 38 | 45 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 9 | restore_sp2 (G=1.1) | restore_sp2 (G=1.1) |  | restore_sp2 (G=1.2) | restore_sp2 (G=1.2) | restore_sp2 (G=1.1) |  |  |  |  |  |  |  |  |  |
| 20 |  | restore_sp2 (G=1.1) |  | restore_sp2 (G=1.2) |  |  | restore_sp2 (G=1.2) |  | restore_sp2 (G=1.3) | restore_sp2 (G=1.1) |  |  |  |  |  |
| 30 |  |  | restore_sp2 (G=1.1) |  |  | restore_sp2 (G=1.2) |  |  | restore_sp2 (G=1.3) |  | restore_sp2 (G=1.3) |  | restore_sp2 (G=1.1) |  |  |
| 50 |  |  |  | restore_sp2 (G=1.2) |  |  |  | restore_sp2 (G=1.2) |  |  |  | restore_sp2 (G=1.3) |  | restore_sp2 (G=1.3) | restore_sp2 (G=1.1) |

## Crossover: remaining steps beyond which waiting for SP=2 beats degrading now

- Wan2.2-TI2V-5B-Diffusers 512x512: remaining > **432** steps (standby=sp1 vs cold SP2 restart 37 s; per-step gap 0.09 s)
- Wan2.2-TI2V-5B-Diffusers 1024x1024: remaining > **439** steps (standby=sp1 vs cold SP2 restart 37 s; per-step gap 0.08 s)
- Wan2.2-TI2V-5B-Diffusers 1536x1536: remaining > **257** steps (standby=sp1 vs cold SP2 restart 37 s; per-step gap 0.14 s)
- Wan2.2-TI2V-5B-Diffusers 832x480x17f: remaining > **303** steps (standby=sp1 vs cold SP2 restart 37 s; per-step gap 0.12 s)
- Wan2.2-TI2V-5B-Diffusers 1280x704x17f: remaining > **134** steps (standby=sp1 vs cold SP2 restart 37 s; per-step gap 0.28 s)
- Wan2.2-TI2V-5B-Diffusers 832x480x49f: remaining > **121** steps (standby=sp1 vs cold SP2 restart 37 s; per-step gap 0.31 s)
- Wan2.2-TI2V-5B-Diffusers 832x480x81f: remaining > **79** steps (standby=sp1 vs cold SP2 restart 37 s; per-step gap 0.47 s)
- Wan2.2-TI2V-5B-Diffusers 832x480x85f: remaining > **66** steps (standby=sp1 vs cold SP2 restart 37 s; per-step gap 0.56 s)
- Wan2.2-TI2V-5B-Diffusers 1280x704x49f: remaining > **48** steps (standby=sp1 vs cold SP2 restart 37 s; per-step gap 0.77 s)
- Wan2.2-TI2V-5B-Diffusers 1280x704x81f: remaining > **25** steps (standby=sp1 vs cold SP2 restart 37 s; per-step gap 1.50 s)
- Z-Image-Turbo 256x256: remaining > **1629** steps (standby=sp1 vs cold SP2 restart 25 s; per-step gap 0.02 s)
- Z-Image-Turbo 512x512: remaining > **1113** steps (standby=sp1 vs cold SP2 restart 25 s; per-step gap 0.02 s)
- Z-Image-Turbo 1024x1024: remaining > **334** steps (standby=sp1 vs cold SP2 restart 25 s; per-step gap 0.07 s)
- Z-Image-Turbo 1536x1536: remaining > **141** steps (standby=sp1 vs cold SP2 restart 25 s; per-step gap 0.18 s)
