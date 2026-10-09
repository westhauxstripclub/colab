# E2 – Fixes (synthetic)

Mean ± s.e. over seeds. *dense-equiv.* = width of a plain MLP reaching the same val loss, in units of one
expert (8 = as good as a dense model with all experts' parameters active; 1 = no better than one expert).
*task info* = MI(routing; true task)/H(task). *expert CKA* = mean pairwise similarity of alive experts' functions.


## Regime E8

| method | val loss | dense-equiv. experts | effective experts | dead | experts per task | task info | useful experts (ablation) | expert CKA |
|---|---|---|---|---|---|---|---|---|
| none | 0.141 ± 0.013 | nan | 6.53 ± 0.40 | 1.25 ± 0.37 | 1.00 ± 0.00 | 0.90 ± 0.03 | 6.58 ± 0.36 | 0.25 ± 0.01 |
| aux_1e-3 | 0.140 ± 0.013 | nan | 6.53 ± 0.40 | 1.25 ± 0.37 | 1.00 ± 0.00 | 0.90 ± 0.03 | 6.58 ± 0.36 | 0.26 ± 0.01 |
| aux_1e-2 | 0.135 ± 0.013 | nan | 6.68 ± 0.38 | 1.12 ± 0.35 | 1.00 ± 0.00 | 0.91 ± 0.03 | 6.71 ± 0.35 | 0.26 ± 0.02 |
| aux_1e-1 | 0.093 ± 0.002 | nan | 7.99 ± 0.00 | 0.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 7.89 ± 0.02 | 0.27 ± 0.02 |
| aux_1 | 0.100 ± 0.002 | nan | 7.99 ± 0.00 | 0.00 ± 0.00 | 1.02 ± 0.00 | 0.99 ± 0.00 | 7.80 ± 0.04 | 0.27 ± 0.01 |
| zloss_1e-3 | 0.145 ± 0.015 | nan | 6.55 ± 0.44 | 1.25 ± 0.41 | 1.00 ± 0.00 | 0.90 ± 0.03 | 6.61 ± 0.40 | 0.26 ± 0.02 |
| aux_1e-2+zloss | 0.139 ± 0.014 | nan | 6.69 ± 0.44 | 1.12 ± 0.40 | 1.00 ± 0.00 | 0.91 ± 0.03 | 6.74 ± 0.41 | 0.25 ± 0.02 |

Dense baselines: 
