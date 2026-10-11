# E5 – Pre-registered held-out test

Pre-registration: `experiments/PREREGISTRATION.md` (committed before any E5 run). Global configs from E4,
10 fresh seeds, test split. Ratio = test loss ÷ best non-oracle, non-dense method in that regime.

| method | E16C16k2 | E32C16 | E8C24 | E16C16zipf08 | XOR8deep2 | CE16 | deep3E16C16 |
|---|---|---|---|---|---|---|---|
| none | 0.133 ± 0.003 | 0.135 ± 0.008 | 0.430 ± 0.005 | 0.169 ± 0.012 | 0.351 ± 0.026 | 0.964 ± 0.021 | 0.073 ± 0.001 |
| aux | 0.127 ± 0.003 | 0.098 ± 0.002 | 0.436 ± 0.006 | 0.141 ± 0.005 | 0.327 ± 0.021 | 0.878 ± 0.015 | 0.071 ± 0.001 |
| zloss | 0.133 ± 0.004 | 0.141 ± 0.006 | 0.434 ± 0.006 | 0.168 ± 0.011 | 0.343 ± 0.022 | 0.969 ± 0.022 | 0.074 ± 0.001 |
| explore_noise | 0.141 ± 0.004 | 0.118 ± 0.005 | 0.465 ± 0.004 | 0.161 ± 0.006 | 0.270 ± 0.004 | 0.901 ± 0.013 | 0.077 ± 0.001 |
| auxfree_bias | 0.129 ± 0.003 | 0.140 ± 0.002 | 0.435 ± 0.005 | 0.207 ± 0.003 | 0.310 ± 0.013 | 0.763 ± 0.008 | 0.072 ± 0.001 |
| expert_choice | 0.126 ± 0.004 | 0.295 ± 0.010 | 0.444 ± 0.005 | 0.277 ± 0.005 | 0.337 ± 0.016 | 0.898 ± 0.025 | 0.074 ± 0.001 |
| sinkhorn | 0.115 ± 0.002 | 0.117 ± 0.004 | 0.432 ± 0.006 | 0.129 ± 0.004 | 0.325 ± 0.015 | 0.818 ± 0.021 | 0.068 ± 0.001 |
| cosine_router | 0.128 ± 0.002 | 0.125 ± 0.008 | 0.437 ± 0.005 | 0.143 ± 0.004 | 0.324 ± 0.011 | 0.883 ± 0.013 | 0.073 ± 0.001 |
| revive_dead | 0.123 ± 0.002 | 0.104 ± 0.002 | 0.435 ± 0.004 | 0.159 ± 0.008 | 0.340 ± 0.020 | 0.766 ± 0.009 | 0.072 ± 0.001 |
| aux_then_release | 0.117 ± 0.003 | 0.103 ± 0.002 | 0.427 ± 0.006 | 0.147 ± 0.004 | 0.302 ± 0.009 | 0.742 ± 0.008 | 0.074 ± 0.001 |
| bias_then_freeze | 0.126 ± 0.002 | 0.095 ± 0.001 | 0.431 ± 0.006 | 0.183 ± 0.012 | 0.320 ± 0.012 | 0.752 ± 0.009 | 0.071 ± 0.001 |
| sinkhorn_then_release | 0.115 ± 0.002 | 0.095 ± 0.002 | 0.432 ± 0.006 | 0.131 ± 0.005 | 0.335 ± 0.020 | 0.809 ± 0.022 | 0.069 ± 0.001 |
| hash_fixed | 0.313 ± 0.009 | 0.304 ± 0.010 | 0.601 ± 0.006 | 0.332 ± 0.010 | 0.545 ± 0.011 | 1.249 ± 0.014 | 0.195 ± 0.005 |
| oracle | 0.096 ± 0.002 | 0.097 ± 0.002 | 0.404 ± 0.006 | 0.097 ± 0.002 | 0.054 ± 0.001 | 0.733 ± 0.008 | 0.066 ± 0.001 |
| dense_active | 0.665 ± 0.004 | 0.776 ± 0.005 | 0.836 ± 0.004 | 0.727 ± 0.009 | 0.505 ± 0.009 | 1.735 ± 0.008 | 0.391 ± 0.004 |
| dense_total | 0.180 ± 0.003 | 0.128 ± 0.002 | 0.471 ± 0.003 | 0.169 ± 0.004 | 0.249 ± 0.004 | 0.955 ± 0.009 | 0.083 ± 0.001 |

## Ratios and robustness

| method | E16C16k2 | E32C16 | E8C24 | E16C16zipf08 | XOR8deep2 | CE16 | deep3E16C16 | geo-mean | worst |
|---|---|---|---|---|---|---|---|---|---|
| sinkhorn_then_release | 1.00 | 1.00 | 1.01 | 1.02 | 1.24 | 1.09 | 1.01 | 1.05 | 1.24 |
| aux_then_release | 1.02 | 1.09 | 1.00 | 1.15 | 1.12 | 1.00 | 1.08 | 1.06 | 1.15 |
| sinkhorn | 1.00 | 1.23 | 1.01 | 1.00 | 1.20 | 1.10 | 1.00 | 1.07 | 1.23 |
| aux | 1.11 | 1.03 | 1.02 | 1.10 | 1.21 | 1.18 | 1.04 | 1.10 | 1.21 |
| bias_then_freeze | 1.10 | 1.00 | 1.01 | 1.42 | 1.18 | 1.01 | 1.04 | 1.10 | 1.42 |
| revive_dead | 1.07 | 1.09 | 1.02 | 1.24 | 1.26 | 1.03 | 1.06 | 1.11 | 1.26 |
| cosine_router | 1.11 | 1.32 | 1.02 | 1.11 | 1.20 | 1.19 | 1.07 | 1.14 | 1.32 |
| explore_noise | 1.23 | 1.24 | 1.09 | 1.26 | 1.00 | 1.21 | 1.13 | 1.16 | 1.26 |
| auxfree_bias | 1.13 | 1.48 | 1.02 | 1.61 | 1.15 | 1.03 | 1.05 | 1.19 | 1.61 |
| none | 1.16 | 1.42 | 1.01 | 1.31 | 1.30 | 1.30 | 1.07 | 1.22 | 1.42 |
| zloss | 1.16 | 1.48 | 1.02 | 1.31 | 1.27 | 1.30 | 1.08 | 1.22 | 1.48 |
| expert_choice | 1.10 | 3.10 | 1.04 | 2.15 | 1.24 | 1.21 | 1.08 | 1.43 | 3.10 |
| hash_fixed | 2.73 | 3.19 | 1.41 | 2.58 | 2.01 | 1.68 | 2.86 | 2.27 | 3.19 |

## Gap closed (0 = dense with active params, 1 = oracle routing)

| method | E16C16k2 | E32C16 | E8C24 | E16C16zipf08 | XOR8deep2 | CE16 | deep3E16C16 |
|---|---|---|---|---|---|---|---|
| sinkhorn_then_release | 0.97 | 1.00 | 0.93 | 0.95 | 0.38 | 0.92 | 0.99 |
| aux_then_release | 0.96 | 0.99 | 0.95 | 0.92 | 0.45 | 0.99 | 0.97 |
| sinkhorn | 0.97 | 0.97 | 0.93 | 0.95 | 0.40 | 0.92 | 0.99 |
| aux | 0.95 | 1.00 | 0.92 | 0.93 | 0.39 | 0.86 | 0.98 |
| bias_then_freeze | 0.95 | 1.00 | 0.94 | 0.86 | 0.41 | 0.98 | 0.98 |
| revive_dead | 0.95 | 0.99 | 0.93 | 0.90 | 0.37 | 0.97 | 0.98 |
| cosine_router | 0.94 | 0.96 | 0.92 | 0.93 | 0.40 | 0.85 | 0.98 |
| explore_noise | 0.92 | 0.97 | 0.86 | 0.90 | 0.52 | 0.83 | 0.97 |
| auxfree_bias | 0.94 | 0.94 | 0.93 | 0.83 | 0.43 | 0.97 | 0.98 |
| none | 0.94 | 0.94 | 0.94 | 0.89 | 0.34 | 0.77 | 0.98 |
| zloss | 0.93 | 0.94 | 0.93 | 0.89 | 0.36 | 0.77 | 0.98 |
| expert_choice | 0.95 | 0.71 | 0.91 | 0.71 | 0.37 | 0.84 | 0.97 |
| hash_fixed | 0.62 | 0.70 | 0.54 | 0.63 | -0.09 | 0.48 | 0.60 |

## Pre-registered predictions

| id | prediction | result | verdict |
|---|---|---|---|
| P1 | aux_then_release has the lowest geo-mean ratio of the 13 methods | lowest: sinkhorn_then_release (1.051); aux_then_release 1.064 | **FAIL** |
| P2 | aux_then_release's worst-case ratio ≤ 1.30 | 1.145 | **PASS** |
| P3 | aux_then_release beats none in ≥ 6 of 7 regimes | 6 of 7 | **PASS** |
| P4 | in E32C16, ≥ 3 of 4 continuous-balance methods are worse than none | worse than none: auxfree_bias, expert_choice | **FAIL** |
| P4 | in E16C16zipf08, ≥ 3 of 4 continuous-balance methods are worse than none | worse than none: auxfree_bias, expert_choice | **FAIL** |
| P5 | E32C16: dead experts (none) within max(25%, 1) of random-owner prediction 19.25 | observed 17.70 | **PASS** |
| P5 | E8C24: dead experts (none) within max(25%, 1) of random-owner prediction 0.32 | observed 0.00 | **PASS** |
| P5 | CE16: dead experts (none) within max(25%, 1) of random-owner prediction 5.70 | observed 5.10 | **PASS** |
