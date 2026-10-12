# E4 – Per-method tuning, then evaluation on fresh seeds and a held-out test split

Tuning: every grid point on the four round-1 regimes, tuning seeds 100–103, scored on the **validation** split.
*Global* = one config per method minimising the geometric-mean ratio to the best config in each regime.
Evaluation: selected configs on fresh seeds 0–9, scored on the **test** split.

| method | global config | geo-mean ratio (tuning) | best per regime: E8 / E32 / E16C8 / E8zipf |
|---|---|---|---|
| none | `-` | 1.417 | `-` / `-` / `-` / `-` |
| aux | `a0.03` | 1.169 | `a0.1` / `a0.3` / `a0.01` / `a0.03` |
| zloss | `z0.0001` | 1.402 | `z0.001` / `z0.0001` / `z0.0001` / `z0.0001` |
| explore_noise | `n1` | 1.393 | `n0.3` / `n1` / `n1` / `n1` |
| auxfree_bias | `u0.1` | 1.309 | `u0.003` / `u0.01` / `u0.1` / `u0.001` |
| expert_choice | `-` | 2.048 | `-` / `-` / `-` / `-` |
| sinkhorn | `i1` | 1.241 | `i10` / `i10` / `i3` / `i1` |
| cosine_router | `s10` | 1.277 | `s10` / `s20` / `s20` / `s5` |
| revive_dead | `r100` | 1.138 | `r50` / `r50` / `r500` / `r500` |
| aux_then_release | `a0.3_t1000` | 1.073 | `a0.1_t1000` / `a0.3_t1000` / `a0.3_t300` / `a0.03_t300` |
| bias_then_freeze | `u0.03_t300` | 1.084 | `u0.03_t100` / `u0.03_t300` / `u0.03_t100` / `u0.03_t100` |
| sinkhorn_then_release | `t1000` | 1.198 | `t300` / `t1000` / `t1000` / `t1000` |
| hash_fixed | `-` | 3.044 | `-` / `-` / `-` / `-` |

## Test loss on evaluation seeds (global config; per-regime-tuned config in brackets when different)

| method | E8 | E32 | E16C8 | E8zipf |
|---|---|---|---|---|
| none | 0.144 ± 0.012 | 0.270 ± 0.005 | 0.106 ± 0.008 | 0.124 ± 0.005 |
| aux | 0.110 ± 0.009 [0.093] | 0.160 ± 0.005 [0.119] | 0.100 ± 0.003 [0.096] | 0.142 ± 0.007 |
| zloss | 0.145 ± 0.012 [0.149] | 0.278 ± 0.007 | 0.107 ± 0.008 | 0.123 ± 0.005 |
| explore_noise | 0.153 ± 0.005 [0.146] | 0.218 ± 0.005 | 0.117 ± 0.012 | 0.125 ± 0.005 |
| auxfree_bias | 0.105 ± 0.002 [0.112] | 0.120 ± 0.001 [0.117] | 0.130 ± 0.004 | 0.216 ± 0.007 [0.212] |
| expert_choice | 0.123 ± 0.010 | 0.224 ± 0.006 | 0.268 ± 0.014 | 0.256 ± 0.010 |
| sinkhorn | 0.101 ± 0.006 [0.094] | 0.188 ± 0.007 [0.166] | 0.121 ± 0.003 [0.125] | 0.137 ± 0.007 |
| cosine_router | 0.146 ± 0.008 | 0.203 ± 0.010 [0.195] | 0.097 ± 0.006 [0.100] | 0.123 ± 0.004 [0.125] |
| revive_dead | 0.104 ± 0.008 [0.100] | 0.154 ± 0.006 [0.137] | 0.100 ± 0.005 [0.099] | 0.140 ± 0.007 [0.132] |
| aux_then_release | 0.093 ± 0.002 [0.093] | 0.114 ± 0.002 | 0.096 ± 0.003 [0.090] | 0.151 ± 0.005 [0.130] |
| bias_then_freeze | 0.105 ± 0.011 [0.096] | 0.108 ± 0.001 | 0.092 ± 0.002 [0.092] | 0.179 ± 0.009 [0.145] |
| sinkhorn_then_release | 0.094 ± 0.002 [0.112] | 0.209 ± 0.009 | 0.092 ± 0.002 | 0.126 ± 0.005 |
| hash_fixed | 0.330 ± 0.018 | 0.469 ± 0.007 | 0.248 ± 0.017 | 0.293 ± 0.011 |
| oracle | 0.092 ± 0.002 | 0.106 ± 0.001 | 0.093 ± 0.001 | 0.100 ± 0.003 |

## Robustness after tuning (global configs, test split)

Ratio = test loss ÷ best tuned method in that regime. *Gap closed* = (L_dense-active − L)/(L_dense-active − L_oracle):
0 = no better than a dense model with one expert's parameters, 1 = as good as routing by the true task.

| method | ratio E8 | ratio E32 | ratio E16C8 | ratio E8zipf | geo-mean | worst | gap closed E8 | gap closed E32 | gap closed E16C8 | gap closed E8zipf |
|---|---|---|---|---|---|---|---|---|---|---|
| aux_then_release | 1.00 | 1.05 | 1.05 | 1.22 | 1.08 | 1.22 | 1.00 | 0.99 | 0.99 | 0.89 |
| bias_then_freeze | 1.13 | 1.00 | 1.00 | 1.45 | 1.13 | 1.45 | 0.98 | 1.00 | 1.00 | 0.82 |
| revive_dead | 1.13 | 1.42 | 1.09 | 1.14 | 1.19 | 1.42 | 0.98 | 0.94 | 0.99 | 0.91 |
| sinkhorn_then_release | 1.01 | 1.93 | 1.00 | 1.03 | 1.19 | 1.93 | 1.00 | 0.86 | 1.00 | 0.94 |
| aux | 1.19 | 1.48 | 1.09 | 1.16 | 1.22 | 1.48 | 0.97 | 0.93 | 0.99 | 0.91 |
| sinkhorn | 1.09 | 1.73 | 1.32 | 1.11 | 1.29 | 1.73 | 0.98 | 0.89 | 0.95 | 0.92 |
| auxfree_bias | 1.13 | 1.11 | 1.42 | 1.75 | 1.33 | 1.75 | 0.98 | 0.98 | 0.94 | 0.74 |
| cosine_router | 1.57 | 1.88 | 1.06 | 1.00 | 1.33 | 1.88 | 0.91 | 0.87 | 0.99 | 0.95 |
| explore_noise | 1.65 | 2.01 | 1.28 | 1.02 | 1.44 | 2.01 | 0.90 | 0.85 | 0.96 | 0.94 |
| none | 1.55 | 2.49 | 1.16 | 1.01 | 1.46 | 2.49 | 0.91 | 0.78 | 0.98 | 0.94 |
| zloss | 1.56 | 2.57 | 1.16 | 1.00 | 1.47 | 2.57 | 0.91 | 0.77 | 0.98 | 0.95 |
| expert_choice | 1.33 | 2.07 | 2.92 | 2.08 | 2.02 | 2.92 | 0.95 | 0.84 | 0.70 | 0.65 |
| hash_fixed | 3.56 | 4.33 | 2.70 | 2.38 | 3.16 | 4.33 | 0.59 | 0.52 | 0.73 | 0.57 |

Rank agreement with the untuned round-1 robustness ranking (geo-mean ratios, 9 shared methods): Spearman ρ = 0.92. Round-1 values: none 1.45, aux 1.19, auxfree_bias 1.33, expert_choice 1.93, sinkhorn 1.27, revive_dead 1.23, aux_then_release 1.10, bias_then_freeze 1.17, sinkhorn_then_release 1.32.

## Dense-equivalent size from a power-law fit (secondary metric)

L(w) = c + a·w^−b fitted to the dense sweep; values are in units of one expert (width 8). `≥` marks an
extrapolation beyond the widest dense model trained (512 = 64 experts' worth); `∞` = below the fitted floor c.

| method | E8 | E32 | E16C8 | E8zipf |
|---|---|---|---|---|
| aux_then_release | 31.7 | ≥233 | 29.6 | 6.0 |
| bias_then_freeze | 25.0 | ≥261 | 32.1 | 4.5 |
| revive_dead | 25.4 | ≥110 | 27.5 | 7.0 |
| sinkhorn_then_release | 31.0 | 52.2 | 32.3 | 8.9 |
| aux | 22.9 | ≥101 | 27.3 | 6.7 |
| sinkhorn | 26.8 | ≥68 | 19.3 | 7.3 |
| auxfree_bias | 25.1 | ≥202 | 16.8 | 3.4 |
| cosine_router | 13.6 | 55.9 | 28.8 | 9.5 |
| explore_noise | 12.4 | 47.5 | 20.5 | 9.2 |
| none | 13.9 | 28.2 | 24.7 | 9.2 |
| zloss | 13.7 | 26.1 | 24.4 | 9.5 |
| expert_choice | 18.7 | 44.2 | 4.4 | 2.7 |
| hash_fixed | 3.0 | 7.3 | 5.0 | 2.3 |

## Which redundancy metrics actually detect redundancy?

Positive control: E16C8 with forced balance (aux 0.1), where each task is split across ~2 experts. Negative
control: E16C8 with free routing, where each task has one surviving expert. Same seeds (paired). A useful
metric must move in the expected direction with a CI excluding 0; *standardised effect* = mean paired
difference ÷ SD of the differences.

| metric | expected (redundant − free) | redundant | free | paired diff [95% CI] | standardised effect | passes |
|---|---|---|---|---|---|---|
| CKA, all probe inputs (round 1) | + | 0.293 ± 0.009 | 0.300 ± 0.011 | -0.007 [-0.024, +0.011] | -0.24 | no |
| CKA, alive experts (round 1) | + | 0.293 ± 0.009 | 0.271 ± 0.014 | +0.021 [+0.001, +0.046] | +0.55 | yes |
| CKA on routed tokens | + | 0.133 ± 0.007 | 0.043 ± 0.006 | +0.090 [+0.083, +0.101] | +5.71 | yes |
| substitution gap | − | 0.729 ± 0.058 | 9.456 ± 0.308 | -8.727 [-9.259, -8.086] | -8.75 | yes |
| useful experts ÷ alive experts (ablation) | − | 0.574 ± 0.081 | 0.976 ± 0.007 | -0.402 [-0.566, -0.257] | -1.52 | yes |
| experts per task | + | 1.975 ± 0.009 | 1.001 ± 0.000 | +0.974 [+0.958, +0.990] | +35.56 | yes |
