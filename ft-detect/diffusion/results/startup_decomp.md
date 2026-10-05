Rank restart decomposition (seconds; from existing logs)

| run | model | T_spawn_s | T_import_s | T_dist_s | T_load_s | T_warmup_s | T_ready_s |
|---|---|---|---|---|---|---|---|
| traj_wan_480p81_sp1_full_20261003-224647 | Wan-AI/Wan2.2-TI2V-5B-Diffusers | 8.7 | 0.0 | 0.07 | 22.0 | 0.7 | 31.46 |
| traj_wan_480p81_sp1_lower_20261003-224755 | Wan-AI/Wan2.2-TI2V-5B-Diffusers | 8.96 | 0.0 | -0.41 | 21.0 | 2.11 | 31.66 |
| traj_wan_480p81_sp1_ref_20261003-224331 | Wan-AI/Wan2.2-TI2V-5B-Diffusers | 8.63 | 0.0 | -0.76 | 22.0 | 1.38 | 31.24 |
| traj_wan_480p81_sp1_ref2_20261003-224510 | Wan-AI/Wan2.2-TI2V-5B-Diffusers | 8.72 | 0.0 | -0.29 | 22.0 | 1.03 | 31.47 |
| traj_wan_480p81_sp2_save_20261003-224221 | Wan-AI/Wan2.2-TI2V-5B-Diffusers | 8.65 | -0.0 | -0.42 | 15.0 | 1.09 | 24.32 |
| traj_zimage_1024_sp1_full_20261003-225152 | Tongyi-MAI/Z-Image-Turbo | 8.79 | 0.0 | -0.13 | 15.0 | 2.88 | 26.55 |
| traj_zimage_1024_sp1_lower_20261003-225239 | Tongyi-MAI/Z-Image-Turbo | 8.63 | 0.0 | -0.17 | 15.0 | 0.9 | 24.36 |
| traj_zimage_1024_sp1_ref_20261003-225019 | Tongyi-MAI/Z-Image-Turbo | 8.61 | 0.0 | -0.14 | 15.0 | 1.17 | 24.64 |
| traj_zimage_1024_sp1_ref2_20261003-225105 | Tongyi-MAI/Z-Image-Turbo | 8.64 | -0.0 | -0.76 | 16.0 | 0.55 | 24.43 |
| traj_zimage_1024_sp2_save_20261003-224929 | Tongyi-MAI/Z-Image-Turbo | 8.64 | -0.0 | -0.22 | 16.0 | 2.19 | 26.6 |
| diff_char_inv_wan22_5b_20261003-194735 | Wan-AI/Wan2.2-TI2V-5B-Diffusers | 8.96 | -0.0 | -0.85 | 16.0 |  |  |
| diff_char_inv_zimage_20261003-194636 | Tongyi-MAI/Z-Image-Turbo | 8.92 | 0.0 | -0.8 | 16.0 |  |  |
| diff_char_wan22_5b_20261003-173105 | Wan-AI/Wan2.2-TI2V-5B-Diffusers | 8.64 | 0.0 | -0.13 | 187.0 |  |  |
| diff_char_wan22_5b_720p_20261003-183106 | Wan-AI/Wan2.2-TI2V-5B-Diffusers | 8.98 | 0.0 | -0.15 | 15.0 |  |  |
| diff_char_zimage_20261003-172311 | Tongyi-MAI/Z-Image-Turbo | 8.78 | 0.0 | -0.8 | 17.0 |  |  |
| diff_char_zimage_conc4 | Wan-AI/Wan2.2-TI2V-5B-Diffusers | 8.7 | 0.0 | -0.77 | 16.0 |  |  |

T_load bundles weight load, host-to-device copy and device setup (not separable in these logs); T_warmup is the server's own warmup requests until /health returns 200.
