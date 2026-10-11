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
