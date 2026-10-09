# MoE load-balancing lab

An instrumented, CPU-runnable testbed for one question: **why do Mixture-of-Experts routers send most tokens to a
few experts, and what turns an MoE into an actual mixture rather than a dense model in disguise?**

It contains a from-scratch MoE layer where every routing/balancing strategy is a swappable option, two
datasets with *known* ground-truth structure (so "specialisation" can be measured, not eyeballed), a battery of
diagnostics, three experiment suites, and the results of running all of them (≈700 synthetic runs plus 18
small language-model runs, all on 4 CPU cores).

<!-- RESULTS -->

---

## 1. Check the premise first

Three parts of the usual story don't hold up, and they change what you should build.

**"Most tokens go to one or two experts" mixes up two different things.**
*Top-k* routing sends **each token** to k experts (k = 1 or 2 in Switch, Mixtral, etc.). That sparsity is the
whole point of an MoE and is not a failure. The failure is **global load imbalance**: across many tokens,
a few experts get most of the traffic and others get almost none. The diagnostic is the *distribution of
load across experts*, not how many experts each token uses. This repo reports it as *effective experts*
(the exponential of the entropy of the load: 8 of 8 = perfectly even, 1 = everything on one expert) and
*dead experts* (experts getting less than 10% of a fair share).

**"Overfitted" is the wrong word, and the measurements point the other way.**
Overfitting means low training loss and a large train→validation gap. A heavily used expert sees *more*
data per parameter, so if anything it should overfit *less*. E1 measures this directly: busy experts have
**higher** training loss on their own tokens (they are capacity-starved, i.e. *underfit*) and a **smaller**
generalisation gap. The few experts that do overfit are the *lightly* used ones. The accurate description is
"over-allocated and starved", not "overfit and underfit".

**Balanced load ≠ a real mixture.**
You can force perfectly even load and still have interchangeable experts. Hash routing does this by
construction, and so does forcing balance with more experts than distinct "skills" in the data. So this repo
measures four separate things: *load* (balance), *specialisation* (mutual information between routing and
ground-truth structure), *redundancy* (CKA similarity between experts' functions) and *utility* (loss
increase when each expert is ablated, plus a dense-equivalent-size score).

## 2. The mechanism: a greedy bandit that only ever tastes what it picks

For a top-1 router with gate `p = softmax(Wx)`, chosen expert `e`, output `y = p_e · E_e(x)`, the router
gradient is

```
∂L/∂logit_j = p_e · (δ_ej − p_j) · ⟨∂L/∂y, E_e(x)⟩
```

Only the **chosen** expert `E_e` appears in it. Every other expert `j` is moved by "how useful `e` was,
times `−p_j`", whatever `j` itself would have done. Unchosen experts also get **zero** parameter gradient. So:

1. **The router is a contextual bandit that only observes the arm it pulled, and it acts greedily (argmax)
   with no exploration.** It never learns that an idle expert might be better somewhere.
2. **Sharing a task is unstable, so each task gets one winner.** If two experts split a latent task, the one
   that is slightly better gets a slightly larger push, takes more of the task, trains more on it, and gets
   better still. Measured: after training, every latent task is served by **1.00–1.02 experts**.
3. **Nothing coordinates *which* expert wins each task.** Winners are decided almost at random, early on,
   so some experts win two or three tasks and others win none. A pure "random owner per task" model predicts
   `E·(1−1/E)^C` dead experts, about 1/e ≈ 37% when there are as many experts as tasks. Observed with 32
   experts and 32 tasks: **38.7%**.
4. **Losing is absorbing.** A loser gets no tokens, so no gradient, so it stays at its random initialisation.
   That makes it worse than every trained expert on every task, so the greedy router never picks it again.
   The prophecy fulfils itself, exactly like dead centroids in k-means or codebook collapse in VQ-VAEs.
5. **Router confidence then locks the result in.** Logit scale grows and router entropy falls towards zero
   over the next few hundred steps, which makes any later switch even less likely.

Steps 1–4 happen **within the first ~100 optimizer steps** (median death step: 10). By then the outcome is
fixed. "Collapse" here is not slow drift over a long run; it's a race that is decided at the start.

## 3. What's in the repo

```
moe_lab/
  moe.py              MoE layer. Every router reduces to a dispatch mask M[N,E] + combine weights G[N,E]:
                      softmax_topk (Switch), topk_renorm (Mixtral), sigmoid_topk (DeepSeek-V3), sinkhorn
                      (S-BASE), expert_choice, hash, oracle; linear or cosine (X-MoE) router; capacity factor;
                      exploration noise; Switch aux loss (batch- or sequence-level); z-loss; aux-loss-free bias
  data.py             synthetic mixture-of-tasks (C clusters × C random teacher nets), and 3-domain byte-level
                      text (Shakespeare / Python stdlib source / CPython reference docs)
  metrics.py          load, router entropy/logit scale, MI specialisation, CKA redundancy, Gini, Spearman
  train_synthetic.py  trainer + interventions (frozen router/experts, router LR, load-normalised grads, SGD,
                      head start, router warm-up, dead-expert revival) + counterfactual diagnostics (loss of
                      every expert on every task, routing regret, per-expert overfitting, ablations)
  train_lm.py         tiny MoE transformer trainer + per-layer/per-domain routing diagnostics
experiments/
  e1_mechanisms.py    WHY: causal interventions on each link of the feedback loop + overfitting test
  e2_fixes.py         FIXES: 15 methods × 3 regimes vs dense-width sweep and oracle routing
  e3_lm.py            LM: do the conclusions survive in a real (tiny) language model?
  analyze.py          figures (results/figures) + tables (results/summary_e*.md) + per-run CSVs
  run_all.sh          everything, resumable
notebooks/moe_load_balancing_lab.ipynb   Colab driver (E3 runs on a GPU there)
tests/test_moe.py     routing-math checks (aux loss = 1 at balance, gradient-flow facts, EC balance, …)
```

### Why a synthetic task with ground truth

In real models you can't tell whether an expert is "specialised" because you don't know what the right
specialisation is. Here, inputs come from C Gaussian clusters and each cluster has its own random teacher
network. One small expert can't fit all C teachers; C specialised experts can. That gives three reference
points:

* **oracle routing** (route by the true cluster id): what perfect specialisation buys;
* **dense MLPs of increasing width**: every MoE result is converted to a *dense-equivalent size*
  ("this MoE is as good as a dense MLP with N experts' worth of parameters all active"). A score near 1
  means the MoE is literally a dense model in disguise;
* **counterfactual losses**: since every expert can be evaluated on every task, you can see which expert
  *would* have been best and how far the router's choice is from it.

### Running it

```bash
pip install -r requirements.txt
python -m pytest -q tests
cd experiments
python e1_mechanisms.py --seeds 8     # ~15 min on 4 CPU cores
python e2_fixes.py --seeds 8          # ~35 min
python e3_lm.py --seeds 2             # ~60 min on CPU; on a GPU: --device cuda --workers 1
python analyze.py                     # figures + tables into results/
```

Results are cached per run, so any script can be re-run to resume. Raw per-run JSON (with full training
curves) goes to `results/<exp>/` and is git-ignored because it's large. `results/runs_*.csv` holds every
run's final metrics. On Colab, open `notebooks/moe_load_balancing_lab.ipynb`.

To try a new idea, add an option to `MoEConfig` / `_route` (or a trainer flag), add a line to `METHODS` in
`e2_fixes.py`, and re-run: only the new runs execute.
