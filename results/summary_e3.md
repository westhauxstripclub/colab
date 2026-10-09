# E3 – Tiny byte-level MoE language model (3 domains)

Val loss in nats/byte, mean ± s.e. over seeds. Per-layer routing stats are averaged over the 2 MoE layers.
*domain info* = MI(top-1 expert; domain)/H(domain). *byte-explained* = MI(top-1 expert; current byte)/H(expert).

*useful experts* = exp(entropy) of each expert's share of the loss increase when it is ablated (per layer).

| method | val loss | shakespeare | python_code | tech_docs | effective experts (of 8) | dead | useful experts | domain info | byte-explained | expert CKA |
|---|---|---|---|---|---|---|---|---|---|---|
| dense_active | 1.659 ± 0.004 | 1.843 ± 0.011 | 1.755 ± 0.004 | 1.410 ± 0.002 | – | – | – | – | – | – |
| dense_total | 1.519 ± 0.015 | 1.705 ± 0.017 | 1.619 ± 0.022 | 1.266 ± 0.007 | – | – | – | – | – | – |
| none | 1.611 ± 0.023 | 1.796 ± 0.018 | 1.713 ± 0.019 | 1.356 ± 0.031 | 4.41 ± 0.41 | 2.0 ± 0.0 | 4.29 ± 0.16 | 0.04 ± 0.00 | 0.77 ± 0.00 | 0.49 ± 0.04 |
| aux_1e-2_batch | 1.560 ± 0.016 | 1.744 ± 0.022 | 1.664 ± 0.007 | 1.307 ± 0.019 | 7.88 ± 0.02 | 0.0 ± 0.0 | 7.24 ± 0.18 | 0.07 ± 0.01 | 0.77 ± 0.02 | 0.58 ± 0.01 |
| aux_1e-2_sequence | 1.559 ± 0.016 | 1.745 ± 0.016 | 1.661 ± 0.009 | 1.305 ± 0.021 | 7.87 ± 0.02 | 0.0 ± 0.0 | 7.18 ± 0.12 | 0.03 ± 0.01 | 0.80 ± 0.02 | 0.58 ± 0.02 |
| aux_1e-2+zloss | 1.576 ± 0.015 | 1.759 ± 0.012 | 1.679 ± 0.018 | 1.323 ± 0.014 | 7.93 ± 0.00 | 0.0 ± 0.0 | 7.31 ± 0.35 | 0.05 ± 0.01 | 0.80 ± 0.00 | 0.56 ± 0.01 |
| auxfree_bias_1e-2 | 1.559 ± 0.025 | 1.746 ± 0.020 | 1.656 ± 0.031 | 1.307 ± 0.025 | 7.98 ± 0.00 | 0.0 ± 0.0 | 7.25 ± 0.33 | 0.09 ± 0.01 | 0.71 ± 0.01 | 0.54 ± 0.02 |
| hash_token | 1.608 ± 0.018 | 1.787 ± 0.011 | 1.715 ± 0.017 | 1.355 ± 0.025 | 7.34 ± 0.00 | 0.0 ± 0.0 | 7.67 ± 0.03 | 0.01 ± 0.00 | 1.00 ± 0.00 | 0.59 ± 0.00 |
| frozen_router | 1.735 ± 0.011 | 1.922 ± 0.012 | 1.817 ± 0.011 | 1.495 ± 0.010 | 5.55 ± 0.80 | 1.5 ± 0.5 | 4.32 ± 0.51 | 0.07 ± 0.03 | 0.27 ± 0.06 | 0.46 ± 0.01 |
