# E1 v2 – Mechanisms with 20 paired seeds

Δ = condition − baseline, paired by seed (same task and initialisation), mean with 95% bootstrap CI; p is a
sign-flip permutation test, Holm-adjusted across the conditions in each regime. **A direction is reported only
when the Holm-adjusted p < 0.05; otherwise "no clear effect"** (the unadjusted CI is shown for size).
`dead` = experts with <10% of a fair share of tokens.


## E8 (8 experts, 8 latent tasks); baseline dead = 1.35 ± 0.18, test loss = 0.149 ± 0.008 (n=20)

| condition | dead | Δ dead vs baseline [95% CI] | p (Holm) | verdict | Δ test loss [95% CI] | n |
|---|---|---|---|---|---|---|
| frozen_router | 0.35 ± 0.13 | -1.00 [-1.45, -0.55] | 0.024 | fewer | +0.235 [+0.214, +0.257] | 20 |
| frozen_experts | 0.00 ± 0.00 | -1.35 [-1.70, -1.00] | 0.001 | fewer | +0.796 [+0.781, +0.812] | 20 |
| identical_init | 1.25 ± 0.20 | -0.10 [-0.55, +0.40] | 1.000 | no clear effect | -0.004 [-0.023, +0.016] | 20 |
| zero_init_output | 0.65 ± 0.13 | -0.70 [-1.10, -0.25] | 0.130 | no clear effect | -0.031 [-0.049, -0.012] | 20 |
| router_warmup_100 | 1.45 ± 0.20 | +0.10 [-0.45, +0.70] | 1.000 | no clear effect | +0.006 [-0.017, +0.030] | 20 |
| router_lr_x0.1 | 1.05 ± 0.20 | -0.30 [-0.70, +0.10] | 1.000 | no clear effect | -0.002 [-0.020, +0.016] | 20 |
| router_lr_x10 | 0.80 ± 0.16 | -0.55 [-1.05, -0.10] | 0.535 | no clear effect | -0.020 [-0.039, -0.003] | 20 |
| adam_load_normalized | 1.30 ± 0.19 | -0.05 [-0.15, +0.00] | 1.000 | no clear effect | -0.002 [-0.005, +0.001] | 20 |
| sgd | 2.30 ± 0.16 | +0.95 [+0.55, +1.30] | 0.008 | more | +0.068 [+0.048, +0.087] | 20 |
| sgd_router_only | 1.75 ± 0.19 | +0.40 [+0.00, +0.85] | 1.000 | no clear effect | +0.018 [-0.001, +0.038] | 20 |
| sgd_experts_only | 1.00 ± 0.18 | -0.35 [-0.70, +0.00] | 1.000 | no clear effect | +0.017 [+0.000, +0.033] | 20 |
| sgd_load_normalized | 1.70 ± 0.19 | +0.35 [+0.05, +0.65] | 0.837 | no clear effect | +0.060 [+0.041, +0.080] | 20 |
| sgd_router_rownorm | 1.10 ± 0.14 | -0.25 [-0.65, +0.20] | 1.000 | no clear effect | +0.013 [-0.005, +0.033] | 20 |
| sgd_rownorm_loadnorm | 0.95 ± 0.14 | -0.40 [-0.75, -0.05] | 0.766 | no clear effect | +0.022 [+0.005, +0.039] | 20 |
| head_start_200 | 1.50 ± 0.17 | +0.15 [-0.30, +0.55] | 1.000 | no clear effect | +0.013 [-0.007, +0.034] | 20 |
| topk_renorm_k1 | 0.35 ± 0.13 | -1.00 [-1.45, -0.55] | 0.024 | fewer | +0.171 [+0.148, +0.194] | 20 |

## E32 (32 experts, 32 latent tasks); baseline dead = 12.30 ± 0.26, test loss = 0.272 ± 0.004 (n=20)

| condition | dead | Δ dead vs baseline [95% CI] | p (Holm) | verdict | Δ test loss [95% CI] | n |
|---|---|---|---|---|---|---|
| frozen_router | 4.45 ± 0.42 | -7.85 [-8.55, -7.15] | 0.001 | fewer | +0.369 [+0.359, +0.379] | 20 |
| frozen_experts | 11.10 ± 0.47 | -1.20 [-2.20, -0.30] | 0.238 | no clear effect | +0.683 [+0.675, +0.691] | 20 |
| identical_init | 14.35 ± 0.43 | +2.05 [+1.20, +2.85] | 0.004 | more | +0.045 [+0.029, +0.059] | 20 |
| zero_init_output | 10.60 ± 0.39 | -1.70 [-2.55, -0.95] | 0.005 | fewer | -0.014 [-0.028, -0.002] | 20 |
| router_warmup_100 | 11.95 ± 0.41 | -0.35 [-1.25, +0.50] | 1.000 | no clear effect | +0.020 [+0.002, +0.037] | 20 |
| router_lr_x0.1 | 12.60 ± 0.38 | +0.30 [-0.55, +1.10] | 1.000 | no clear effect | +0.087 [+0.070, +0.103] | 20 |
| router_lr_x10 | 11.55 ± 0.39 | -0.75 [-1.65, +0.20] | 0.981 | no clear effect | -0.024 [-0.037, -0.009] | 20 |
| adam_load_normalized | 12.20 ± 0.28 | -0.10 [-0.40, +0.20] | 1.000 | no clear effect | +0.002 [-0.002, +0.007] | 20 |
| sgd | 24.30 ± 0.29 | +12.00 [+11.20, +12.80] | 0.001 | more | +0.405 [+0.391, +0.418] | 20 |
| sgd_router_only | 18.90 ± 0.40 | +6.60 [+5.85, +7.30] | 0.001 | more | +0.155 [+0.142, +0.168] | 20 |
| sgd_experts_only | 12.70 ± 0.29 | +0.40 [-0.20, +0.95] | 1.000 | no clear effect | +0.212 [+0.202, +0.223] | 20 |
| sgd_load_normalized | 22.65 ± 0.22 | +10.35 [+9.70, +11.00] | 0.001 | more | +0.531 [+0.517, +0.544] | 20 |
| sgd_router_rownorm | 15.50 ± 0.51 | +3.20 [+2.35, +3.95] | 0.001 | more | +0.413 [+0.400, +0.425] | 20 |
| sgd_rownorm_loadnorm | 13.65 ± 0.41 | +1.35 [+0.80, +1.95] | 0.005 | more | +0.487 [+0.478, +0.497] | 20 |
| head_start_200 | 12.75 ± 0.47 | +0.45 [-0.45, +1.40] | 1.000 | no clear effect | +0.027 [+0.012, +0.042] | 20 |
| topk_renorm_k1 | 4.45 ± 0.42 | -7.85 [-8.55, -7.15] | 0.001 | fewer | +0.196 [+0.185, +0.207] | 20 |

## Where does SGD's extra collapse come from? (2×2 factorial + probes)

Main effects are averaged over the other factor, per seed, then bootstrapped. Row-norm gives every expert's
router row an equal-norm gradient; load-norm rescales each expert's gradient by mean load / its load.

| router opt. | expert opt. | extra | dead E8 | dead E32 | test loss E8 | test loss E32 |
|---|---|---|---|---|---|---|
| Adam | Adam |  | 1.35 ± 0.18 | 12.30 ± 0.26 | 0.149 ± 0.008 | 0.272 ± 0.004 |
| SGD | Adam |  | 1.75 ± 0.19 | 18.90 ± 0.40 | 0.167 ± 0.010 | 0.427 ± 0.007 |
| Adam | SGD |  | 1.00 ± 0.18 | 12.70 ± 0.29 | 0.166 ± 0.006 | 0.484 ± 0.006 |
| SGD | SGD |  | 2.30 ± 0.16 | 24.30 ± 0.29 | 0.217 ± 0.010 | 0.676 ± 0.007 |
| SGD | SGD | load-norm experts | 1.70 ± 0.19 | 22.65 ± 0.22 | 0.209 ± 0.012 | 0.802 ± 0.007 |
| SGD | SGD | row-norm router | 1.10 ± 0.14 | 15.50 ± 0.51 | 0.162 ± 0.007 | 0.685 ± 0.008 |
| SGD | SGD | both | 0.95 ± 0.14 | 13.65 ± 0.41 | 0.171 ± 0.008 | 0.759 ± 0.006 |

| regime | main effect of SGD on the router (Δ dead) | main effect of SGD on the experts (Δ dead) | interaction |
|---|---|---|---|
| E8 | +0.85 [+0.55, +1.15] | +0.10 [-0.10, +0.33] | +0.90 [+0.30, +1.50] |
| E32 | +9.10 [+8.47, +9.68] | +2.90 [+2.45, +3.38] | +5.00 [+3.95, +6.00] |

## Ownership and timing (no balancing, 20 seeds)

| regime | dead (observed) | random-owner prediction E·(1−1/E)^C | experts per task | median death step | deaths before step 100 | n |
|---|---|---|---|---|---|---|
| E8 | 1.35 ± 0.18 | 2.75 | 1.00 ± 0.00 | 10 | 100% | 20 |
| E16C8 | 8.50 ± 0.14 | 9.55 | 1.00 ± 0.00 | 10 | 100% | 20 |
| E32 | 12.30 ± 0.26 | 11.59 | 1.03 ± 0.01 | 10 | 99% | 20 |

## Is a popular expert overfit? (20 seeds)

| regime | ρ(load, own train loss) | ρ(load, val − train gap) | gap, experts owning 1 task | gap, experts owning ≥2 tasks | n |
|---|---|---|---|---|---|
| E8__baseline | 0.49 ± 0.08 | -0.07 ± 0.08 | 0.006 ± 0.001 | 0.007 ± 0.001 | 20 |
| overfit__baseline | 0.67 ± 0.05 | -0.61 ± 0.07 | 0.107 ± 0.003 | 0.072 ± 0.003 | 20 |
