# MoE load-balancing lab

An instrumented testbed that runs on a CPU, built for one question: **why do Mixture-of-Experts routers send most tokens to a
few experts, and what turns an MoE into an actual mixture rather than a dense model in disguise?**

The repo has five parts:
- A from-scratch MoE layer in which every routing and balancing strategy is a swappable option.
- Two datasets with *known* ground-truth structure, so "specialisation" can be measured rather than eyeballed.
- A set of diagnostics.
- Three experiment suites.
- The results of running all of them: ~800 synthetic runs and 18 small language-model runs, all on 4 CPU cores.

## TL;DR

1. **It isn't overfitting.** Busy experts fit their own tokens *worse* (they are capacity-starved) and generalise
   *better*. Idle experts aren't underfit so much as *never trained*. The few experts that do overfit are the lightly
   used ones.
2. **Collapse is a race that is decided in the first ~100 optimizer steps** (median death step: 10). It is not
   slow drift over a long run.
3. **The cause is a greedy router that only receives feedback about the expert it picked.**
   - Each latent task ends up owned by exactly one expert (1.00–1.02 experts per task).
   - Which expert wins a task is effectively random and uncoordinated across tasks.
   - Losers get no gradient ever again, so they can never come back.

   With 32 experts and 32 tasks, 39% of experts die. A model that gives each task a uniformly random owner
   predicts 36% (≈1/e).
4. **What didn't matter, and what did:**
   - *No effect:* initial expert asymmetry (identical init), the router's learning rate (×0.1 or ×10), giving one
     expert a head start, and a router warm-up.
   - *Prevents it:* freezing the router.
   - *Makes it much worse:* plain SGD (77% of experts dead vs 39% with Adam). Adam's per-parameter normalisation
     is already a partial fix.
5. **The standard fixes work, but only if they are strong enough.** The textbook Switch aux loss at α=0.01 barely
   helps here. α=0.1, DeepSeek-V3-style bias balancing, or Sinkhorn routing reach near-oracle quality when the data
   has one skill per expert. z-loss does nothing for balance: it *increased* collapse with 32 experts.
6. **Perfect balance is the wrong target when the data isn't shaped like the experts.** Two cases:
   - *More experts than skills:* every method that actually enforces balance does worse than no balancing at all.
     It ends up splitting each skill across redundant experts.
   - *Skill frequencies follow a Zipf distribution:* forcing balance does worse than no balancing by up to 73%.
     Even the oracle routing is imbalanced there (5.5 effective experts out of 8).
7. **The most robust recipe here is to balance hard while the race is decided, then release.** With a strong aux
   loss for the first 300 steps then turned off, val loss is within 0–26% of the best method in every regime.
   No other method stays within 30% everywhere.
8. **Tiny language model:** see [E3](#e3--does-it-hold-in-a-real-tiny-language-model).

<!-- E3-TLDR -->

---

## 1. Check the premise first

Three parts of the usual story don't hold up, and they change what you should build.

**"Most tokens go to one or two experts" mixes up two different things.**
*Top-k* routing sends **each token** to k experts (k = 1 or 2 in Switch, Mixtral, etc.). That sparsity is the
whole point of an MoE and is not a failure. The failure is **global load imbalance**: across many tokens, a few
experts get most of the traffic and others get almost none. The diagnostic is the *distribution of load across
experts*, not how many experts each token uses. This repo reports it in two ways:
- *Effective experts*: the exponential of the entropy of the load. 8 of 8 means perfectly even; 1 means everything
  is on one expert.
- *Dead experts*: experts that get less than 10% of a fair share of tokens.

Well-tuned production MoEs (with an aux loss or bias balancing) generally don't collapse in this crude sense. Their
open problem is whether the experts that *are* used actually specialise.

**"Overfitted" is the wrong word.**
Overfitting means low training loss and a large gap between training and validation loss. A heavily used expert
sees *more* data per parameter, so if anything it should overfit *less*. [E1](#is-a-popular-expert-overfit)
measures this directly and finds the reverse of the overfitting story. The accurate description is "over-allocated
and starved", not "overfit and underfit".

**Balanced load ≠ a real mixture.**
You can force perfectly even load and still have interchangeable experts. Hash routing does this by construction,
and so does forcing balance onto data with fewer distinct "skills" than experts. So this repo measures four separate
things:
- *Load*: how evenly tokens are spread.
- *Specialisation*: mutual information between routing and the ground-truth structure.
- *Utility*: how much the loss rises when each expert is removed.
- *Dense-equivalent size*: how wide a plain MLP must be to match the MoE.

## 2. The mechanism: a greedy bandit that only ever tastes what it picks

For a top-1 router with gate `p = softmax(Wx)`, chosen expert `e`, and output `y = p_e · E_e(x)`, the router
gradient is

```
∂L/∂logit_j = p_e · (δ_ej − p_j) · ⟨∂L/∂y, E_e(x)⟩
```

Only the **chosen** expert `E_e` appears in it. Every other expert `j` is moved by "how useful `e` was, times
`−p_j`", whatever `j` itself would have done. Unchosen experts also get **zero** parameter gradient. From there:

1. **The router is a contextual bandit with no exploration.** It acts greedily (argmax) and only observes the arm
   it pulled, so it never learns that an idle expert might be better somewhere.
2. **Sharing a task is unstable, so each task gets one winner.** Suppose two experts split a latent task. The
   slightly better one gets a slightly larger push, takes more of the task, trains more on it, and gets better
   still. Measured: after training, every latent task is served by **1.00–1.02 experts**.
3. **Nothing coordinates which expert wins each task.** Some experts win two or three tasks and others win none.
   - If owners were assigned at random, about `E·(1−1/E)^C` experts would end up dead.
   - Observed vs that prediction: **12.4 vs 11.6** (32 experts, 32 tasks); **8.4 vs 9.6** (16 experts, 8 tasks);
     **1.3 vs 2.8** (8 experts, 8 tasks; small systems do somewhat better than random).
   - With 32 experts, freezing the experts at their random init *still* kills 11.4. The greedy, uncoordinated
     choice is the core of the problem; expert learning amplifies it and makes it permanent.
4. **Losing is absorbing.** A loser gets no tokens, so no gradient, so it stays at its random initialisation. That
   makes it worse than every trained expert on every task, so the greedy router never picks it again. It is a
   self-fulfilling prophecy, exactly like dead centroids in k-means or codebook collapse in VQ-VAEs
   (see the counterfactual heat-map below).
5. **Router confidence then locks the result in.** Logit scale grows and router entropy falls towards zero over
   the next ~1000 steps, which makes any later switch even less likely.

**Why the textbook aux loss is often too weak.** The Switch loss is `L_aux = α·E·Σ f_i·P_i`, where `f_i` is the
fraction of tokens dispatched to expert `i` (not differentiable) and `P_i` is its mean router probability. Its
gradient on a logit is

```
∂L_aux/∂logit_i ∝ P_i · (f_i − Σ_j f_j P_j)
```

For a dead expert (`f_i = 0`), the restoring push is proportional to its own router probability `P_i`.
- Once the router is confident (`P_i ≈ 0`), the aux loss can no longer revive that expert
  (`tests/test_moe.py::test_aux_loss_pull_on_a_dead_expert_vanishes_with_its_probability`).
- The loss has to win the race in the first ~100 steps, so `α` must be large relative to the task-loss gradient at
  that point. `α` is not scale-free: a value tuned for one loss scale is weak or strong for another.

## 3. Results

All synthetic numbers are means over 8 seeds (± standard error). Each seed changes both the task and the
initialisation. The regimes are:

| regime | experts | latent tasks | task frequencies | what it represents |
|---|---|---|---|---|
| E8 | 8 | 8 | uniform | the textbook case: one skill per expert |
| E32 | 32 | 32 | uniform | same, at a size where collapse is severe |
| E16C8 | 16 | 8 | uniform | more experts than distinct skills |
| E8zipf | 8 | 8 | Zipf (s = 1.2) | skewed skill frequencies, like token frequencies in text |

Each expert is a 16→8→8 MLP. Each task's teacher is a random 16→32→8 network. A dense MLP with the active
parameters of one expert (width 8) reaches val loss 0.66 on E8, versus 0.09 for oracle routing, so specialisation
matters a lot in this task.

### E1 – why does it happen?

![Token share per expert over training](results/figures/e1_dynamics.png)

Left: a typical run without balancing. Experts 0, 5 and 6 lose all their tokens within ~50 steps, long before the
router becomes confident (middle) or its logits grow (right). Note the log-scaled time axis.

![Counterfactual loss of each expert on each task](results/figures/e1_expert_task_loss.png)

This shows what *each* expert would score on *each* task if it handled that task alone; the orange box marks the
expert the router actually uses.
- At step 0 every expert is equally mediocre everywhere.
- By step 200 each task has one owner that has improved on it.
- The dead experts (rows 0, 5 and 6) stay at their initial loss on *every* task for the rest of training. They were
  never worse by nature; they are worse because they were never trained.

![Interventions on the feedback loop](results/figures/e1_mechanisms_dead_experts.png)

| intervention (no balancing anywhere) | dead / 8 (E8) | dead / 32 (E32) | reading |
|---|---|---|---|
| baseline | 1.25 ± 0.37 | 12.4 ± 0.4 | |
| freeze router (random fixed routing) | 0 | 4.5 ± 0.8 (= dead at init) | no *new* deaths without a learning router |
| freeze experts (router learns to pick among random experts) | 0 | 11.4 ± 0.6 | at scale, greedy selection alone is enough |
| identical expert init | 1.12 ± 0.23 | 13.9 ± 0.8 | initial expert asymmetry is not needed |
| expert output layer initialised to 0 | 0.75 ± 0.25 | 10.4 ± 0.8 | small help: less noise for the router to react to |
| router frozen for first 100 steps | 1.38 ± 0.32 | 11.8 ± 0.4 | no help: deaths just move to right after step 100 (median death step 110 in E32, vs 10) |
| router LR ×0.1 / ×10 | 1.25 / 0.75 | 12.3 / 11.9 | no consistent effect |
| Adam + per-expert grads rescaled by 1/load | 1.25 ± 0.37 | 12.5 ± 0.5 | identical, as Adam's scale invariance predicts |
| **SGD** | **2.50 ± 0.19** | **24.5 ± 0.5** | update size ∝ token count: strong amplifier |
| SGD + per-expert grads rescaled by 1/load | 1.88 ± 0.40 | 22.4 ± 0.4 | only part of SGD's amplification comes from expert updates |
| expert 0 trained alone for 200 steps first | 1.00 ± 0.19 | 12.5 ± 0.5 | no takeover (expert 0 ends with 17% of tokens vs a fair 12.5%) |
| Mixtral-style renormalised gate with k = 1 | 0 | 4.5 ± 0.8 | the gate is ≡ 1, so the router gets **no** gradient: an implementation pitfall |

Timing and ownership (no balancing):

| regime | dead | random-owner prediction | experts per task | median death step | deaths before step 100 |
|---|---|---|---|---|---|
| E8 | 1.25 ± 0.37 | 2.75 | 1.00 | 10 | 100% |
| E16C8 | 8.38 ± 0.18 | 9.55 | 1.00 | 10 | 100% |
| E32 | 12.38 ± 0.37 | 11.59 | 1.02 | 10 | 99% |

#### Is a popular expert overfit?

![Overfitting test](results/figures/e1_overfitting_test.png)

Each point is one expert in one run. The orange points come from a data-scarce variant: 100 examples per task and
4× wider experts, so overfitting is actually possible there.

| setting | ρ(load, own train loss) | ρ(load, val − train gap) | gap: experts owning 1 task | gap: experts owning ≥2 tasks |
|---|---|---|---|---|
| plentiful data (1000/task) | +0.43 ± 0.12 | −0.35 ± 0.07 | 0.005 | 0.006 |
| scarce data (100/task) | +0.69 ± 0.10 | −0.64 ± 0.13 | 0.112 | 0.070 |

Busier experts have *higher* training loss and a *smaller* generalisation gap. "Overfitting" has the sign wrong.

### E2 – what fixes it, and is balance even the right target?

The table below summarises 18 methods across the four regimes. Each cell is val loss divided by the best
non-oracle method in that regime (1.00 = best there). Rows are sorted by geometric mean. Full per-regime tables,
covering dead experts, effective experts, task information, ablation utility and dense-equivalent size, are in
[`results/summary_e2.md`](results/summary_e2.md).

| method | E8 | E32 | E16C8 | E8zipf | geo-mean | worst |
|---|---|---|---|---|---|---|
| **aux α=0.1 for 300 steps, then off** | 1.14 | 1.26 | **1.00** | 1.04 | **1.10** | **1.26** |
| aux-free bias for 300 steps, then frozen | **1.00** | 1.07 | 1.31 | 1.33 | 1.17 | 1.33 |
| aux α=0.1 (Switch loss) | 1.02 | 1.09 | 1.38 | 1.33 | 1.19 | 1.38 |
| revive dead experts by splitting the busiest | 1.24 | 1.42 | 1.23 | 1.06 | 1.23 | 1.42 |
| Sinkhorn selection (S-BASE) | 1.02 | 1.39 | 1.39 | 1.31 | 1.27 | 1.39 |
| Sinkhorn for 300 steps, then plain top-k | 1.28 | 2.27 | 1.03 | 1.03 | 1.32 | 2.27 |
| aux-free bias (DeepSeek-V3) | 1.28 | **1.00** | 1.43 | 1.73 | 1.33 | 1.73 |
| aux α=0.01 (the common default) | 1.48 | 2.03 | 1.09 | 1.04 | 1.36 | 2.03 |
| no balancing | 1.54 | 2.33 | 1.23 | **1.00** | 1.45 | 2.33 |
| expert-choice routing | 1.25 | 1.86 | 3.06 | 1.95 | 1.93 | 3.06 |

Methods run only on E8 and E32 (val loss, E8 / E32): z-loss 1e-3 0.145 / **0.346** (worse than no balancing,
0.141 / 0.270); aux 1e-2 + z-loss 0.139 / 0.302; exploration noise 0.154 / 0.221; cosine router 0.143 / 0.209;
hash routing 0.321 / 0.462; oracle 0.092 / 0.106.

![E32 methods](results/figures/e2_fixes_E32.png)

**When the data has one equally common skill per expert (E8, E32), balancing is the cure.**
- Strong aux loss, bias balancing and Sinkhorn all remove the dead experts and land close to oracle routing. In E32
  the best of them (bias, 0.116) beats a dense MLP with *all 32 experts' parameters active* (0.226 at width 256),
  using one expert's compute per token.
- Even the unbalanced MoE (0.270) is far from "dense in disguise" in compute terms: it matches a dense MLP about 26
  experts wide. What it gives up is the factor of 2.3 in loss that the fixes recover.
- Exploration noise and the cosine router help partially. z-loss is a stabiliser for logit magnitude, not a
  balancer: here it made collapse worse.

![E16C8 methods](results/figures/e2_fixes_E16C8.png)

**When there are more experts than skills (E16C8), balance just splits each skill across two experts.**
- With no balancing, exactly one expert per task survives (8 dead) at loss 0.109.
- Forcing balance keeps all 16 alive but at 1.97 experts per task, and it is *worse*:
  - Switch aux α=0.1: 0.122
  - Sinkhorn: 0.123
  - aux-free bias: 0.126
  - expert choice: 0.271
- The ablation column shows the redundancy. Removing one twin of a pair costs little, so only ~6–10 of the 16
  balanced experts are "useful".
- Balancing early and then releasing gives the best result of any method (0.089). Early on, every expert gets
  trained; then the best-trained candidate for each skill wins and the spares die off.

![E8zipf methods](results/figures/e2_fixes_E8zipf.png)

**When skill frequencies are skewed (E8zipf), the optimal routing is imbalanced, and forcing balance hurts.**
- Oracle routing uses only 5.5 effective experts out of 8.
- Every method that keeps forcing balance is worse than doing nothing (0.129): Switch aux α=0.1 0.171, Sinkhorn
  0.169, aux-free bias 0.222, expert choice 0.251.
- The damage concentrates on the rarest tasks (loss 0.29 → 0.33–0.45). Their dedicated experts get diluted with
  overflow from the common task.
- The release variants (0.132–0.133) avoid that penalty.
- Nothing closes the gap to the oracle (0.100). The remaining loss is mostly on the rare tasks, which lose the race
  to share an expert.

**Release is not free.**
- Sinkhorn-then-release falls straight back to 12 dead experts in E32. Sinkhorn balanced the *selection* while the
  router's own logits stayed collapsed, and removing the mask exposes that.
- After an aux loss is released, the router keeps most of what it learned (3.6 dead, vs 12.4 with no balancing and
  0.4 with continuous aux).
- After a frozen bias, the router keeps nearly all of it (1.25 dead), but a frozen bias also keeps forcing balance
  on Zipf data.

### E3 – does it hold in a real (tiny) language model?

<!-- E3-RESULTS -->

## 4. What I'd actually do

1. **Instrument from step 0.** The outcome is decided in the first ~1% of training, so a load chart from step 1000
   onward tells you nothing about *why*. Log these per layer:
   - effective experts and dead experts;
   - router entropy and logit scale;
   - ablation utility (dropping each expert in turn).

   Compare every MoE against dense baselines at matched *active* and *total* parameters.
2. **Balance the learning signal, not the load.** The thing worth guaranteeing is that every expert gets enough
   tokens *early* to become competent. Here that means a strong balance term (aux α≈0.1 relative to an O(1) task
   loss, or bias balancing) during warm-up, then annealing it.
   - For production, `α` has to be calibrated against the task-loss gradient, not copied.
   - Permanent strict balance trades quality for even load whenever the data isn't uniformly structured.
3. **Know which constraint you are paying for.** In production, balance is also a *systems* requirement: expert
   parallelism, capacity limits and token dropping all assume roughly even load. That is a real reason to keep
   some balance pressure, but it is a throughput argument, not a quality one. This lab doesn't model throughput.
4. **Avoid known traps:**
   - SGD on routers.
   - Renormalised top-1 gates (no router gradient at all).
   - Treating z-loss as a balancer.
   - Treating Sinkhorn or expert-choice balance as evidence the router itself is healthy.
   - Expert choice in autoregressive models (it uses future tokens).
5. **If you have more experts than distinct skills, dead experts aren't the disease.** Fewer, larger experts (or a
   shared expert for common features) can beat forcing redundant experts to split each skill.

## 5. Limits of this evidence (what would change these conclusions)

- **The synthetic task is a single MoE layer on fixed inputs**, with linearly separable tasks and MSE loss.
  - Deep models add a pathway these runs can't show: upstream layers can *move token representations* towards a
    favoured expert (the representation collapse described by Chi et al., 2022).
  - E3 probes this only at toy scale (2 layers, 8 experts, byte-level, 2500 steps, 2 seeds).
- **Hyperparameters were not tuned per method:**
  - aux α was swept over {1e-3, 1e-2, 1e-1, 1};
  - the bias rate (1e-2), noise std (1.0) and release step (300) were set once, the last from E1's death timing.
  - A method that looks mediocre here may be fine when tuned.
- **The balance-then-release recipe was proposed *after* seeing E2's main results** and then tested on the same four
  regimes. It is a hypothesis that survived one test, not one validated on held-out settings. The most useful next
  experiment is to test it on new regimes and in a deeper model.
- **8 seeds per cell.** Differences under ~2 standard errors (e.g. the router-LR and zero-init rows) should be read
  as "no clear effect".
- **"Dense-equivalent size" saturates in E8**, because wide dense MLPs nearly solve that task.
- **CKA similarity between experts was computed but did not separate methods** (0.13–0.34 everywhere). The
  ablation-based "useful experts" count turned out to be the informative redundancy measure.
- The SGD result implies something besides expert-update size amplifies collapse under SGD. My guess is the
  router's own un-normalised updates, but that is **untested**.
- **What would most change my mind:** large-scale evidence that collapse mainly happens late in training, or is
  driven by representation drift rather than an early per-skill race. In that case early-phase interventions like
  balance-then-release would be the wrong lever.

## 6. What's in the repo

```
moe_lab/
  moe.py              MoE layer. Every router reduces to a dispatch mask M[N,E] + combine weights G[N,E]:
                      softmax_topk (Switch), topk_renorm (Mixtral), sigmoid_topk (DeepSeek-V3), sinkhorn
                      (S-BASE), expert_choice, hash, oracle; linear or cosine (X-MoE) router; capacity factor;
                      exploration noise; Switch aux loss (batch- or sequence-level); z-loss; aux-loss-free bias
  data.py             synthetic mixture-of-tasks (C clusters × C random teacher nets), and 3-domain byte-level
                      text (Shakespeare / Python stdlib source / CPython reference docs)
  metrics.py          load, router entropy/logit scale, MI specialisation, CKA, Gini, Spearman
  train_synthetic.py  trainer + interventions (frozen router/experts, router LR and warm-up, load-normalised
                      grads, SGD, head start, zero-init, balance-then-release, dead-expert revival) + counterfactual
                      diagnostics (every expert on every task, routing regret, per-expert overfitting, ablations)
  train_lm.py         tiny MoE transformer trainer + per-layer / per-domain routing diagnostics
experiments/
  e1_mechanisms.py    WHY: causal interventions on each link of the feedback loop + overfitting test
  e2_fixes.py         FIXES: 18 methods × 4 regimes vs a dense-width sweep and oracle routing
  e3_lm.py            LM: do the conclusions survive in a real (tiny) language model?
  analyze.py          figures (results/figures) + tables (results/summary_e*.md) + per-run CSVs
  run_all.sh          everything, resumable
notebooks/moe_load_balancing_lab.ipynb   Colab driver (E3 runs on a GPU there)
tests/test_moe.py     routing-math checks (aux loss = 1 at balance, gradient-flow facts, EC balance, …)
results/              summary tables, figures, one CSV row per run (raw JSON is git-ignored)
```

**Why a synthetic task with ground truth.** In real models you can't tell whether an expert is "specialised",
because you don't know what the right specialisation is. Here, inputs come from C Gaussian clusters and each
cluster has its own random teacher network. One small expert can't fit all C teachers; C specialised experts can.
That gives three reference points:
- **Oracle routing** (route by the true cluster id): what perfect specialisation buys.
- **Dense MLPs of increasing width:** every MoE result is converted into a *dense-equivalent size*. A score near
  one expert means the MoE is literally a dense model in disguise.
- **Counterfactual losses:** every expert is evaluated on every task, so you can see which expert *would* have
  been best and how far the router's choice is from it.

**Running it:**

```bash
pip install -r requirements.txt
python -m pytest -q tests
cd experiments
python e1_mechanisms.py --seeds 8     # ~20 min on 4 CPU cores
python e2_fixes.py --seeds 8          # ~50 min
python e3_lm.py --seeds 2             # ~1.5 h on CPU; on a GPU: --device cuda --workers 1
python analyze.py                     # figures + tables into results/
```

Results are cached per run, so any script can be re-run to resume.
- Raw per-run JSON, with full training curves, goes to `results/<exp>/` and is git-ignored because it is large.
- `results/runs_*.csv` holds every run's final metrics.
- On Colab, open `notebooks/moe_load_balancing_lab.ipynb`.

To try a new idea, add an option to `MoEConfig` / `_route` (or a trainer flag), add a line to `METHODS` in
`e2_fixes.py`, and re-run: only the new runs execute.

## References

- Shazeer et al., 2017. *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer* (noisy top-k, importance loss).
- Lepikhin et al., 2020. *GShard*.
- Fedus, Zoph & Shazeer, 2021. *Switch Transformers* (the `E·Σ f·P` aux loss, capacity factor).
- Lewis et al., 2021. *BASE Layers* (balanced assignment). Roller et al., 2021. *Hash Layers for Large Sparse Models*.
- Clark et al., 2022. *Unified Scaling Laws for Routed Language Models* (Sinkhorn-based S-BASE).
- Zhou et al., 2022. *Mixture-of-Experts with Expert Choice Routing*.
- Zoph et al., 2022. *ST-MoE: Designing Stable and Transferable Sparse Expert Models* (router z-loss).
- Chi et al., 2022. *On the Representation Collapse of Sparse Mixture of Experts* (X-MoE cosine router).
- Dai et al., 2024. *DeepSeekMoE* (fine-grained and shared experts). Jiang et al., 2024. *Mixtral of Experts* (routing analysis).
- Wang et al., 2024. *Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts*; DeepSeek-AI, 2024. *DeepSeek-V3 Technical Report*.
- Muennighoff et al., 2024. *OLMoE*.
- Qiu et al., 2025. *Demons in the Detail: On Implementing Load Balancing Loss for Training Specialized Mixture-of-Expert Models* (global- vs micro-batch balancing).
