# E1 – Mechanisms of collapse (synthetic)

Mean ± standard error over seeds. `dead` = experts receiving <10% of a fair share of tokens.


## Regime E8 (8 experts, 8 latent tasks)

| condition | dead experts | dead at step 0 | effective experts | experts per task | task info recovered | val loss | n |
|---|---|---|---|---|---|---|---|
| baseline | 1.25 ± 0.37 | 0.00 ± 0.00 | 6.53 ± 0.40 | 1.00 ± 0.00 | 0.90 ± 0.03 | 0.141 ± 0.013 | 8 |
| frozen_router | 0.00 ± 0.00 | 0.00 ± 0.00 | 6.94 ± 0.13 | 2.27 ± 0.10 | 0.58 ± 0.02 | 0.380 ± 0.006 | 8 |
| frozen_experts | 0.00 ± 0.00 | 0.00 ± 0.00 | 7.06 ± 0.11 | 2.06 ± 0.11 | 0.62 ± 0.03 | 0.944 ± 0.004 | 8 |
| identical_init | 1.12 ± 0.23 | 0.00 ± 0.00 | 6.61 ± 0.26 | 1.00 ± 0.00 | 0.90 ± 0.02 | 0.137 ± 0.012 | 8 |
| router_lr_x0.1 | 1.25 ± 0.37 | 0.00 ± 0.00 | 6.52 ± 0.40 | 1.00 ± 0.00 | 0.89 ± 0.03 | 0.153 ± 0.017 | 8 |
| router_lr_x10 | 0.75 ± 0.31 | 0.00 ± 0.00 | 7.10 ± 0.36 | 1.00 ± 0.00 | 0.94 ± 0.03 | 0.126 ± 0.013 | 8 |
| adam_load_normalized | 1.25 ± 0.37 | 0.00 ± 0.00 | 6.54 ± 0.40 | 1.00 ± 0.00 | 0.90 ± 0.03 | 0.140 ± 0.013 | 8 |
| sgd | 2.50 ± 0.19 | 0.00 ± 0.00 | 5.13 ± 0.19 | 1.01 ± 0.01 | 0.78 ± 0.02 | 0.227 ± 0.015 | 8 |
| sgd_load_normalized | 1.88 ± 0.40 | 0.00 ± 0.00 | 5.86 ± 0.41 | 1.00 ± 0.00 | 0.84 ± 0.03 | 0.224 ± 0.024 | 8 |
| head_start_200 | 1.00 ± 0.19 | 0.00 ± 0.00 | 6.74 ± 0.22 | 1.00 ± 0.00 | 0.92 ± 0.02 | 0.140 ± 0.008 | 8 |
| topk_renorm_k1 | 0.00 ± 0.00 | 0.00 ± 0.00 | 6.94 ± 0.13 | 2.27 ± 0.10 | 0.58 ± 0.02 | 0.313 ± 0.008 | 8 |

## Regime E32 (32 experts, 32 latent tasks)

| condition | dead experts | dead at step 0 | effective experts | experts per task | task info recovered | val loss | n |
|---|---|---|---|---|---|---|---|
| baseline | 12.38 ± 0.37 | 4.50 ± 0.78 | 18.15 ± 0.37 | 1.02 ± 0.01 | 0.83 ± 0.01 | 0.270 ± 0.006 | 8 |
| frozen_router | 4.50 ± 0.78 | 4.50 ± 0.78 | 21.85 ± 0.62 | 2.91 ± 0.09 | 0.62 ± 0.01 | 0.634 ± 0.008 | 8 |
| frozen_experts | 11.38 ± 0.56 | 4.50 ± 0.78 | 17.60 ± 0.58 | 1.76 ± 0.03 | 0.68 ± 0.01 | 0.955 ± 0.002 | 8 |
| identical_init | 13.88 ± 0.83 | 4.50 ± 0.78 | 16.33 ± 0.73 | 1.04 ± 0.01 | 0.80 ± 0.01 | 0.311 ± 0.014 | 8 |
| router_lr_x0.1 | 12.25 ± 0.45 | 4.50 ± 0.78 | 18.24 ± 0.46 | 1.10 ± 0.02 | 0.82 ± 0.01 | 0.350 ± 0.008 | 8 |
| router_lr_x10 | 11.88 ± 0.79 | 4.50 ± 0.78 | 18.33 ± 0.75 | 1.01 ± 0.00 | 0.83 ± 0.01 | 0.257 ± 0.011 | 8 |
| adam_load_normalized | 12.50 ± 0.46 | 4.50 ± 0.78 | 17.91 ± 0.45 | 1.02 ± 0.01 | 0.83 ± 0.01 | 0.272 ± 0.008 | 8 |
| sgd | 24.50 ± 0.46 | 4.50 ± 0.78 | 6.65 ± 0.33 | 1.24 ± 0.03 | 0.50 ± 0.02 | 0.686 ± 0.010 | 8 |
| sgd_load_normalized | 22.38 ± 0.37 | 4.50 ± 0.78 | 8.62 ± 0.34 | 1.31 ± 0.05 | 0.57 ± 0.01 | 0.801 ± 0.012 | 8 |
| head_start_200 | 12.50 ± 0.50 | 4.50 ± 0.78 | 17.22 ± 0.53 | 1.03 ± 0.01 | 0.81 ± 0.01 | 0.292 ± 0.010 | 8 |
| topk_renorm_k1 | 4.50 ± 0.78 | 4.50 ± 0.78 | 21.85 ± 0.62 | 2.91 ± 0.09 | 0.62 ± 0.01 | 0.463 ± 0.010 | 8 |

## When do experts die, and how many would a random owner per task predict?

| regime | dead (observed) | random-owner prediction E·(1−1/E)^C | experts per task | median death step | deaths before step 100 | n |
|---|---|---|---|---|---|---|
| E8 | 1.25 ± 0.37 | 2.75 | 1.00 ± 0.00 | 10 | 100% | 8 |
| E32 | 12.38 ± 0.37 | 11.59 | 1.02 ± 0.01 | 10 | 99% | 8 |

Head start: expert 0 (trained alone for 200 steps first) ends with 0.17 ± 0.02 of tokens (fair share 0.125).

## Is a popular expert *overfit*?

Per-expert loss on the tokens routed to it. If popular experts were overfitting, their train/val
gap would grow with load. Spearman correlations are computed per run, then averaged.

| regime | ρ(load, train loss) | ρ(load, val−train gap) | mean gap, experts serving 1 task | mean gap, experts serving ≥2 tasks | n |
|---|---|---|---|---|---|
| E8__baseline | 0.43 ± 0.12 | -0.35 ± 0.07 | 0.005 ± 0.001 | 0.006 ± 0.001 | 8 |
| overfit__baseline | 0.69 ± 0.10 | -0.64 ± 0.13 | 0.112 ± 0.005 | 0.070 ± 0.003 | 8 |
