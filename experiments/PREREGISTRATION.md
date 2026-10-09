# Pre-registration: held-out test of the round-1 conclusions (E5) and the depth experiment (E6)

**Status:** written and committed *before* any E5 or E6 run. The commit that adds this file is the timestamp.
Results will be reported against these exact predictions and rules, pass or fail
(`results/summary_e5.md`, `results/summary_e6.md`, generated mechanically by `experiments/analyze2.py`).

## Why

Round 1 proposed "balance hard early, then release" *after* seeing the results it was then tested on, and its
hyperparameters were not tuned per method. Round 2 tuned every method (E4, on the four round-1 regimes only) and
now freezes those choices and tests them on settings that played no role in designing or tuning anything.

## What is frozen

- **Configs:** each method's *global* config from E4 (`results/tuned_configs.json`, selected on the validation
  split of tuning seeds 100–103). They are copied below; nothing is re-tuned on held-out data.
- **Held-out regimes** (`experiments/e5_heldout.py`; none was used in round 1 or in E4):

  | id | experts | tasks | what is new |
  |---|---|---|---|
  | E16C16k2 | 16 | 16 | top-2 routing (everything before was top-1) |
  | E32C16 | 32 | 16 | twice as many experts as skills, at larger scale |
  | E8C24 | 8 | 24 | fewer experts than skills |
  | E16C16zipf08 | 16 | 16 | milder Zipf skew (s = 0.8) at a different size |
  | XOR8deep2 | 8 per layer | 8 | tasks not linearly separable; 2 MoE layers on learned features (dense pre-block) |
  | CE16 | 16 | 16 | cross-entropy classification instead of regression |
  | deep3E16C16 | 16 per layer | 16 | 3 stacked MoE layers on learned representations |

- **Seeds and split:** evaluation seeds 0–9; every number is the mean **test**-split loss over seeds.
- **Methods (13):** none, aux, zloss, explore_noise, auxfree_bias, expert_choice, sinkhorn, cosine_router,
  revive_dead, aux_then_release, bias_then_freeze, sinkhorn_then_release, hash_fixed.
  References (not ranked): oracle routing, dense with the active parameters, dense with all parameters.
- **Metric:** ratio = method's test loss ÷ the best of the 13 methods in that regime; summarised by the
  geometric mean over the 7 regimes and the worst case.

## Predictions (E5)

| id | prediction | how it is scored |
|---|---|---|
| P1 | `aux_then_release` has the lowest geometric-mean ratio of the 13 methods | exact ranking |
| P2 | `aux_then_release`'s worst-case ratio is ≤ 1.30 | max over the 7 regimes |
| P3 | `aux_then_release` beats `none` in at least 6 of the 7 regimes | mean test loss per regime |
| P4 | In E32C16 and in E16C16zipf08 (each scored separately), at least 3 of the 4 continuous-balance methods (aux, auxfree_bias, sinkhorn, expert_choice) are worse than `none` | mean test loss |
| P5 | Without balancing, the dead-expert count matches the random-owner model E·(1−1/E)^C to within max(25%, 1 expert) in E32C16 (predicts 19.3), E8C24 (0.3) and CE16 (5.7), each scored separately | mean over seeds |

## Predictions (E6, depth and the representation pathway)

E6 (`experiments/e6_deep.py`, 8 seeds, same frozen configs where a method is tuned) is partly exploratory; these are
the directional claims it is meant to test:

| id | prediction | how it is scored |
|---|---|---|
| D1 | In deep models on linearly separable tasks (depth 2 and depth 4, each scored separately), freezing the router does **not** prevent new dead experts: dead per layer at the end > dead at step 0 | paired 95% CI excludes 0 |
| D2 | Detaching the router's input (no routing gradient into the representations) **reduces** dead experts relative to `none` (depth 2 and depth 4, each scored separately) | paired 95% CI below 0 |
| D3 | On XOR tasks without a dense pre-block, learned routing recovers < 0.3 of the task information in the last MoE layer | mean over seeds |

## What would count against round 1

- P1/P2/P3 failing means balance-then-release was overfit to the round-1 regimes; the README must say so.
- P4 failing means "continuous balance hurts when experts > skills or skills are skewed" does not generalise.
- P5 failing means the random-owner picture of collapse is incomplete beyond the regimes it was fitted on.
- D1 passing and D2 failing would mean collapse in deep models comes mostly from incidental representation drift,
  not from the routing gradient pulling representations, so detaching is not a fix.

## Frozen configs (from E4, verbatim)

<!-- FROZEN-CONFIGS -->
