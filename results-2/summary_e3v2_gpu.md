# E3 v2 – Scaled byte-level MoE LM (e3v2_lm_gpu)

6 layers, d_model 256, 32 experts × 128 (top-2), 10000 steps. Mean ± s.e. over seeds;
routing stats averaged over MoE layers. *late deaths* = experts dead at the end whose last alive evaluation was
after 10% of training (step 1000). *revived* = experts that were dead at some evaluation and alive at the end.

| method | val loss | shakespeare | python_code | tech_docs | effective experts | dead / layer | useful experts | domain info | byte-explained | late deaths / run | revived / run | n |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dense_active | 1.603 ± 0.004 | 1.628 ± 0.014 | 1.834 ± 0.017 | 1.390 ± 0.011 | – | – | – | – | – | – | – | 3 |
| dense_total | 3.433 ± 0.037 | 4.029 ± 0.014 | 3.530 ± 0.099 | 2.809 ± 0.011 | – | – | – | – | – | – | – | 3 |
| none | 1.768 ± 0.025 | 1.695 ± 0.032 | 2.112 ± 0.021 | 1.549 ± 0.024 | 9.60 ± 0.39 | 15.50 ± 0.54 | 7.64 ± 0.45 | 0.15 ± 0.02 | 0.53 ± 0.04 | 17.00 | 59.67 | 3 |
| aux_1e-2 | 2.459 ± 0.008 | 2.533 ± 0.005 | 2.772 ± 0.025 | 2.133 ± 0.007 | 31.30 ± 0.06 | 0.00 ± 0.00 | 19.53 ± 0.44 | 0.15 ± 0.00 | 0.55 ± 0.00 | 0.00 | 14.00 | 3 |
| aux_1e-2_then_release | 2.451 ± 0.004 | 2.518 ± 0.014 | 2.762 ± 0.014 | 2.131 ± 0.014 | 22.06 ± 0.37 | 5.22 ± 0.24 | 18.95 ± 0.26 | 0.15 ± 0.01 | 0.63 ± 0.03 | 31.33 | 12.00 | 3 |
| frozen_router | 1.556 ± 0.025 | 1.572 ± 0.017 | 1.813 ± 0.048 | 1.328 ± 0.026 | 6.32 ± 0.26 | 22.44 ± 0.15 | 4.77 ± 0.21 | 0.13 ± 0.00 | 0.27 ± 0.02 | 47.00 | 31.67 | 3 |
| detach_input | 1.766 ± 0.039 | 1.708 ± 0.035 | 2.070 ± 0.048 | 1.566 ± 0.036 | 9.61 ± 0.65 | 16.56 ± 0.24 | 8.44 ± 0.21 | 0.15 ± 0.01 | 0.57 ± 0.03 | 15.00 | 52.00 | 3 |

## Best checkpoint vs final checkpoint

*best val* = lowest val loss over all evaluations (selected on the same val split, so slightly optimistic, equally
for every method); routing stats are at that checkpoint.

**Overfitting:** val loss bottoms out at a median step of 3000 of 10000 and rises afterwards, so the final-checkpoint losses above measure how much each model overfits rather than how well it models the text. Compare methods on *best val* here.

| method | best val | step of best (per seed) | final val | final − best | effective experts at best | dead / layer at best | n |
|---|---|---|---|---|---|---|---|
| dense_active | 1.341 ± 0.008 | 4000, 3500, 3500 | 1.603 ± 0.004 | +0.263 | – | – | 3 |
| dense_total | 1.347 ± 0.004 | 2000, 1500, 1500 | 3.433 ± 0.037 | +2.086 | – | – | 3 |
| none | 1.354 ± 0.009 | 3500, 3500, 3000 | 1.768 ± 0.025 | +0.414 | 8.76 ± 0.40 | 17.44 ± 0.24 | 3 |
| aux_1e-2 | 1.340 ± 0.006 | 2000, 2500, 2000 | 2.459 ± 0.008 | +1.119 | 30.71 ± 0.12 | 0.00 ± 0.00 | 3 |
| aux_1e-2_then_release | 1.342 ± 0.004 | 2000, 2500, 2000 | 2.451 ± 0.004 | +1.109 | 25.98 ± 0.44 | 2.28 ± 0.29 | 3 |
| frozen_router | 1.350 ± 0.014 | 3500, 4500, 4000 | 1.556 ± 0.025 | +0.206 | 6.43 ± 0.38 | 21.11 ± 0.29 | 3 |
| detach_input | 1.358 ± 0.007 | 3000, 3500, 3500 | 1.766 ± 0.039 | +0.408 | 9.29 ± 0.62 | 17.39 ± 0.45 | 3 |

## Which pathway moves the router logits? (none / frozen router / detached input, final checkpoint)

| method | rms change from reps | rms change from router weights | rep share | dead if only reps moved | dead if only router moved | dead (actual, probe) |
|---|---|---|---|---|---|---|
| none | 0.40 ± 0.01 | 1.88 ± 0.11 | 0.06 ± 0.01 | 3.72 ± 0.20 | 3.89 ± 0.28 | 15.78 ± 0.87 |
| aux_1e-2 | 0.41 ± 0.00 | 1.92 ± 0.02 | 0.05 ± 0.00 | 2.78 ± 0.39 | 0.89 ± 0.06 | 0.00 ± 0.00 |
| frozen_router | 0.86 ± 0.01 | 0.00 ± 0.00 | 1.00 ± 0.00 | 22.67 ± 0.10 | 0.06 ± 0.06 | 22.67 ± 0.10 |
| detach_input | 0.40 ± 0.01 | 2.10 ± 0.08 | 0.04 ± 0.01 | 2.78 ± 0.39 | 5.94 ± 1.22 | 17.11 ± 0.34 |
