"""Round-2 analysis: paired statistics, tuning, held-out pre-registered test, depth, scaled LM.

Writes results/summary_e1v2.md, summary_e4.md, summary_e5.md, summary_e6.md, summary_e3v2.md and figures. Every
comparison between conditions is paired by seed (same task and initialisation) with bootstrap CIs; tables with
several comparisons use Holm-adjusted p-values.
"""
from __future__ import annotations

import json
import math

import numpy as np
from analyze import (FIG, INK2, OUT, SEQ, SERIES, death_steps, experts_per_task, fmt, group, mean_se, plt,
                     random_owner_prediction)
from common import RESULTS, load_results

from moe_lab.stats import compare, holm

V = lambda r, k: r["final"].get(k, float("nan"))  # noqa: E731


def seedmap(runs, fn):
    return {r["config"]["seed"]: fn(r) for r in runs}


def ci_str(c, digits=2):
    return f"{c['diff']:+.{digits}f} [{c['ci_lo']:+.{digits}f}, {c['ci_hi']:+.{digits}f}]"


def verdict(c, p_adj, lower_is="fewer", higher_is="more"):
    """Direction only when the Holm-adjusted p < 0.05 (the unadjusted CI is shown alongside)."""
    if p_adj >= 0.05:
        return "no clear effect"
    return higher_is if c["diff"] > 0 else lower_is


# ===================================================================================================== E1 v2
E1_ORDER = ["baseline", "frozen_router", "frozen_experts", "identical_init", "zero_init_output", "router_warmup_100",
            "router_lr_x0.1", "router_lr_x10", "adam_load_normalized", "sgd", "sgd_router_only", "sgd_experts_only",
            "sgd_load_normalized", "sgd_router_rownorm", "sgd_rownorm_loadnorm", "head_start_200", "topk_renorm_k1"]


def analyze_e1v2():
    res = load_results("e1v2_mechanisms")
    if not res:
        return
    g = group(res)
    L = ["# E1 v2 – Mechanisms with 20 paired seeds\n",
         "Δ = condition − baseline, paired by seed (same task and initialisation), mean with 95% bootstrap CI; p is a",
         "sign-flip permutation test, Holm-adjusted across the conditions in each regime. **A direction is reported only",
         "when the Holm-adjusted p < 0.05; otherwise \"no clear effect\"** (the unadjusted CI is shown for size).",
         "`dead` = experts with <10% of a fair share of tokens.\n"]
    forest = {}
    for regime, E in (("E8", 8), ("E32", 32)):
        base = g.get(f"{regime}__baseline")
        if not base:
            continue
        b_dead = seedmap(base, lambda r: V(r, "dead_experts"))
        b_test = seedmap(base, lambda r: V(r, "test_loss"))
        rows, comps = [], []
        for cond in E1_ORDER[1:]:
            runs = g.get(f"{regime}__{cond}")
            if not runs:
                continue
            c = compare(seedmap(runs, lambda r: V(r, "dead_experts")), b_dead)
            ct = compare(seedmap(runs, lambda r: V(r, "test_loss")), b_test)
            rows.append((cond, runs, c, ct))
            comps.append(c["p"])
        adj = holm(comps) if comps else []
        L += [f"\n## {regime} ({E} experts, {E} latent tasks); baseline dead = {fmt(*mean_se([V(r, 'dead_experts') for r in base]), 2)}, "
              f"test loss = {fmt(*mean_se([V(r, 'test_loss') for r in base]))} (n={len(base)})\n",
              "| condition | dead | Δ dead vs baseline [95% CI] | p (Holm) | verdict | Δ test loss [95% CI] | n |",
              "|---|---|---|---|---|---|---|"]
        forest[regime] = []
        for (cond, runs, c, ct), p in zip(rows, adj):
            L.append(f"| {cond} | {fmt(*mean_se([V(r, 'dead_experts') for r in runs]), 2)} | {ci_str(c)} | {p:.3f} | "
                     f"{verdict(c, p)} | {ci_str(ct, 3)} | {c['n']} |")
            forest[regime].append((cond, c, p))
    _forest_plot(forest)
    L += _sgd_factorial(g)
    L += _ownership(g)
    L += _overfitting(g)
    (OUT / "summary_e1v2.md").write_text("\n".join(L) + "\n")


def _forest_plot(forest):
    if not forest:
        return
    fig, axes = plt.subplots(1, len(forest), figsize=(6 * len(forest), 0.33 * len(E1_ORDER) + 1.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (regime, rows) in zip(axes, forest.items()):
        y = np.arange(len(rows))[::-1]
        for yi, (cond, c, p) in zip(y, rows):
            color = SERIES[0] if p < 0.05 else INK2
            ax.plot([c["ci_lo"], c["ci_hi"]], [yi, yi], color=color, lw=2)
            ax.plot(c["diff"], yi, "o", color=color, ms=5)
        ax.axvline(0, color=INK2, lw=1)
        ax.set_yticks(y, [r[0] for r in rows])
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Δ dead experts vs baseline (paired, 95% CI)")
        ax.set_title(f"{regime}", loc="left")
    fig.suptitle("Interventions on the feedback loop, 20 paired seeds (blue = Holm-adjusted p < 0.05)", x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "e1v2_forest.png", dpi=160)
    plt.close(fig)


def _sgd_factorial(g):
    L = ["\n## Where does SGD's extra collapse come from? (2×2 factorial + probes)\n",
         "Main effects are averaged over the other factor, per seed, then bootstrapped. Row-norm gives every expert's",
         "router row an equal-norm gradient; load-norm rescales each expert's gradient by mean load / its load.\n",
         "| router opt. | expert opt. | extra | dead E8 | dead E32 | test loss E8 | test loss E32 |", "|---|---|---|---|---|---|---|"]
    cells = [("Adam", "Adam", "", "baseline"), ("SGD", "Adam", "", "sgd_router_only"), ("Adam", "SGD", "", "sgd_experts_only"),
             ("SGD", "SGD", "", "sgd"), ("SGD", "SGD", "load-norm experts", "sgd_load_normalized"),
             ("SGD", "SGD", "row-norm router", "sgd_router_rownorm"), ("SGD", "SGD", "both", "sgd_rownorm_loadnorm")]
    for ro, eo, extra, cond in cells:
        cols = []
        for key in ("dead_experts", "test_loss"):
            for regime in ("E8", "E32"):
                runs = g.get(f"{regime}__{cond}", [])
                cols.append(fmt(*mean_se([V(r, key) for r in runs]), 2 if key == "dead_experts" else 3))
        L.append(f"| {ro} | {eo} | {extra} | " + " | ".join(cols) + " |")
    L += ["", "| regime | main effect of SGD on the router (Δ dead) | main effect of SGD on the experts (Δ dead) | interaction |",
          "|---|---|---|---|"]
    from moe_lab.stats import bootstrap_ci

    for regime in ("E8", "E32"):
        m = {c: seedmap(g.get(f"{regime}__{c}", []), lambda r: V(r, "dead_experts"))
             for c in ("baseline", "sgd_router_only", "sgd_experts_only", "sgd")}
        seeds = sorted(set.intersection(*[set(v) for v in m.values()])) if all(m.values()) else []
        if not seeds:
            continue
        a, r_, e_, b = (np.array([m[c][s] for s in seeds]) for c in ("baseline", "sgd_router_only", "sgd_experts_only", "sgd"))
        router_eff = bootstrap_ci(((r_ - a) + (b - e_)) / 2)
        expert_eff = bootstrap_ci(((e_ - a) + (b - r_)) / 2)
        inter = bootstrap_ci((b - e_) - (r_ - a))
        f = lambda t: f"{t[0]:+.2f} [{t[1]:+.2f}, {t[2]:+.2f}]"  # noqa: E731
        L.append(f"| {regime} | {f(router_eff)} | {f(expert_eff)} | {f(inter)} |")
    return L


def _ownership(g):
    L = ["\n## Ownership and timing (no balancing, 20 seeds)\n",
         "| regime | dead (observed) | random-owner prediction E·(1−1/E)^C | experts per task | median death step | deaths before step 100 | n |",
         "|---|---|---|---|---|---|---|"]
    for regime, E, C in (("E8", 8, 8), ("E16C8", 16, 8), ("E32", 32, 32)):
        runs = g.get(f"{regime}__baseline", [])
        if not runs:
            continue
        ds = [x for r in runs for x in death_steps(r)]
        L.append(f"| {regime} | {fmt(*mean_se([V(r, 'dead_experts') for r in runs]), 2)} | {random_owner_prediction(E, C):.2f} | "
                 f"{fmt(*mean_se([experts_per_task(r) for r in runs]), 2)} | {np.median(ds) if ds else float('nan'):.0f} | "
                 f"{np.mean(np.array(ds) < 100) if ds else float('nan'):.0%} | {len(runs)} |")
    return L


def _overfitting(g):
    from moe_lab.metrics import spearman

    L = ["\n## Is a popular expert overfit? (20 seeds)\n",
         "| regime | ρ(load, own train loss) | ρ(load, val − train gap) | gap, experts owning 1 task | gap, experts owning ≥2 tasks | n |",
         "|---|---|---|---|---|---|"]
    for regime in ("E8__baseline", "overfit__baseline"):
        runs = g.get(regime, [])
        if not runs:
            continue
        rt, rg, g1, g2 = [], [], [], []
        for r in runs:
            f = r["final"]
            n_tasks = (np.array(f["cluster_to_expert"]).argmax(1)[:, None] == np.arange(len(f["load"]))).sum(0)
            tok, tr, gap = (np.array(f[k], float) for k in ("per_expert_train_tokens", "per_expert_train_loss", "per_expert_gap"))
            ok = ~np.isnan(tr) & ~np.isnan(gap)
            if ok.sum() >= 3:
                rt.append(spearman(tok[ok], tr[ok]))
                rg.append(spearman(tok[ok], gap[ok]))
            g1 += gap[ok & (n_tasks == 1)].tolist()
            g2 += gap[ok & (n_tasks >= 2)].tolist()
        L.append(f"| {regime} | {fmt(*mean_se(rt), 2)} | {fmt(*mean_se(rg), 2)} | {fmt(*mean_se(g1))} | {fmt(*mean_se(g2))} | {len(runs)} |")
    return L


# ===================================================================================================== E4
TUNE_REGIMES = ["E8", "E32", "E16C8", "E8zipf"]
METHODS = ["none", "aux", "zloss", "explore_noise", "auxfree_bias", "expert_choice", "sinkhorn", "cosine_router",
           "revive_dead", "aux_then_release", "bias_then_freeze", "sinkhorn_then_release", "hash_fixed"]


def power_law_dense_equivalent(widths, losses, expert_width=8, max_width=512):
    """Fit L(w) = c + a * w^-b to the dense sweep; return an inverse mapping loss -> width (in expert units)."""
    w, l = np.asarray(widths, float), np.asarray(losses, float)
    best = None
    for c in np.linspace(0, l.min() * 0.999, 200):
        X = np.log(w)
        Y = np.log(l - c)
        A = np.vstack([np.ones_like(X), -X]).T
        coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
        err = ((A @ coef - Y) ** 2).sum()
        if best is None or err < best[0]:
            best = (err, c, math.exp(coef[0]), coef[1])
    _, c, a, b = best

    def inv(loss):
        if loss <= c:
            return float("inf"), True
        width = (a / (loss - c)) ** (1 / b)
        return width / expert_width, width > max_width

    return inv


def analyze_e4():
    tuned_path = RESULTS / "tuned_configs.json"
    if not tuned_path.exists():
        return
    tuned = json.loads(tuned_path.read_text())
    L = ["# E4 – Per-method tuning, then evaluation on fresh seeds and a held-out test split\n",
         "Tuning: every grid point on the four round-1 regimes, tuning seeds 100–103, scored on the **validation** split.",
         "*Global* = one config per method minimising the geometric-mean ratio to the best config in each regime.",
         "Evaluation: selected configs on fresh seeds 0–9, scored on the **test** split.\n",
         "| method | global config | geo-mean ratio (tuning) | best per regime: " + " / ".join(TUNE_REGIMES) + " |", "|---|---|---|---|"]
    for m in METHODS:
        if m in tuned:
            t = tuned[m]
            L.append(f"| {m} | `{t['global']['tag']}` | {t['global']['geo_ratio']:.3f} | "
                     + " / ".join(f"`{t['per_regime'][r]['tag']}`" for r in TUNE_REGIMES) + " |")
    _sensitivity_plot(tuned)

    g = group(load_results("e4_eval"))
    if not g:
        (OUT / "summary_e4.md").write_text("\n".join(L) + "\n")
        return
    test = lambda name: [V(r, "test_loss") for r in g.get(name, [])]  # noqa: E731
    L += ["\n## Test loss on evaluation seeds (global config; per-regime-tuned config in brackets when different)\n",
          "| method | " + " | ".join(TUNE_REGIMES) + " |", "|" + "---|" * (len(TUNE_REGIMES) + 1)]
    for m in METHODS + ["oracle"]:
        cells = []
        for reg in TUNE_REGIMES:
            name = f"{reg}__{m}__global" if m != "oracle" else f"{reg}__oracle__-"
            s = fmt(*mean_se(test(name)))
            if g.get(f"{reg}__{m}__regime"):
                s += f" [{np.mean(test(f'{reg}__{m}__regime')):.3f}]"
            cells.append(s)
        L.append(f"| {m} | " + " | ".join(cells) + " |")

    # Robustness ranking + gap closed.
    best = {reg: min(np.mean(test(f"{reg}__{m}__global")) for m in METHODS if g.get(f"{reg}__{m}__global"))
            for reg in TUNE_REGIMES}
    dense_active = {reg: np.mean(test(f"{reg}__dense_8__-")) for reg in TUNE_REGIMES}
    oracle = {reg: np.mean(test(f"{reg}__oracle__-")) for reg in TUNE_REGIMES}
    rows = []
    for m in METHODS:
        if not all(g.get(f"{reg}__{m}__global") for reg in TUNE_REGIMES):
            continue
        losses = [np.mean(test(f"{reg}__{m}__global")) for reg in TUNE_REGIMES]
        ratios = [l / best[reg] for l, reg in zip(losses, TUNE_REGIMES)]
        gaps = [(dense_active[reg] - l) / (dense_active[reg] - oracle[reg]) for l, reg in zip(losses, TUNE_REGIMES)]
        rows.append((float(np.exp(np.mean(np.log(ratios)))), max(ratios), m, ratios, gaps))
    L += ["\n## Robustness after tuning (global configs, test split)\n",
          "Ratio = test loss ÷ best tuned method in that regime. *Gap closed* = (L_dense-active − L)/(L_dense-active − L_oracle):",
          "0 = no better than a dense model with one expert's parameters, 1 = as good as routing by the true task.\n",
          "| method | ratio " + " | ratio ".join(TUNE_REGIMES) + " | geo-mean | worst | gap closed " + " | gap closed ".join(TUNE_REGIMES) + " |",
          "|" + "---|" * (2 * len(TUNE_REGIMES) + 3)]
    for gm, worst, m, ratios, gaps in sorted(rows):
        L.append(f"| {m} | " + " | ".join(f"{x:.2f}" for x in ratios) + f" | {gm:.2f} | {worst:.2f} | "
                 + " | ".join(f"{x:.2f}" for x in gaps) + " |")
    L += _round1_rank_comparison(rows)
    L += _dense_equivalent_table(g, test, rows)
    L += _metric_validation(g)
    (OUT / "summary_e4.md").write_text("\n".join(L) + "\n")


def _round1_rank_comparison(rows):
    r1 = {"none": 1.45, "aux": 1.19, "auxfree_bias": 1.33, "expert_choice": 1.93, "sinkhorn": 1.27, "revive_dead": 1.23,
          "aux_then_release": 1.10, "bias_then_freeze": 1.17, "sinkhorn_then_release": 1.32}  # round 1, untuned (E2)
    common = [(r[2], r[0]) for r in rows if r[2] in r1]
    if len(common) < 3:
        return []
    from moe_lab.metrics import spearman

    rho = spearman([r1[m] for m, _ in common], [gm for _, gm in common])
    return [f"\nRank agreement with the untuned round-1 robustness ranking (geo-mean ratios, {len(common)} shared methods): "
            f"Spearman ρ = {rho:.2f}. Round-1 values: " + ", ".join(f"{m} {r1[m]:.2f}" for m, _ in common) + "."]


def _dense_equivalent_table(g, test, rows):
    L = ["\n## Dense-equivalent size from a power-law fit (secondary metric)\n",
         "L(w) = c + a·w^−b fitted to the dense sweep; values are in units of one expert (width 8). `≥` marks an",
         "extrapolation beyond the widest dense model trained (512 = 64 experts' worth); `∞` = below the fitted floor c.\n",
         "| method | " + " | ".join(TUNE_REGIMES) + " |", "|" + "---|" * (len(TUNE_REGIMES) + 1)]
    invs = {}
    for reg in TUNE_REGIMES:
        ws = [8, 16, 32, 64, 128, 256, 512]
        ls = [np.mean(test(f"{reg}__dense_{w}__-")) for w in ws]
        if all(np.isfinite(ls)):
            invs[reg] = power_law_dense_equivalent(ws, ls)
    for _, _, m, _, _ in sorted(rows):
        cells = []
        for reg in TUNE_REGIMES:
            if reg not in invs:
                cells.append("–")
                continue
            v, extrap = invs[reg](np.mean(test(f"{reg}__{m}__global")))
            cells.append("∞" if math.isinf(v) else (f"≥{v:.0f}" if extrap else f"{v:.1f}"))
        L.append(f"| {m} | " + " | ".join(cells) + " |")
    return L


def _metric_validation(g):
    red, free = g.get("E16C8__control_redundant__-", []), g.get("E16C8__control_free__-", [])
    if not red or not free:
        return []

    def useful_frac(r):
        alive = len(r["final"]["load"]) - r["final"]["dead_experts"]
        return r["final"]["useful_experts"] / max(alive, 1)

    metrics = [
        ("CKA, all probe inputs (round 1)", lambda r: V(r, "all_expert_cka_mean"), "+"),
        ("CKA, alive experts (round 1)", lambda r: V(r, "alive_expert_cka_mean"), "+"),
        ("CKA on routed tokens", lambda r: V(r, "routed_cka"), "+"),
        ("substitution gap", lambda r: V(r, "substitution_gap"), "−"),
        ("useful experts ÷ alive experts (ablation)", useful_frac, "−"),
        ("experts per task", experts_per_task, "+"),
    ]
    L = ["\n## Which redundancy metrics actually detect redundancy?\n",
         "Positive control: E16C8 with forced balance (aux 0.1), where each task is split across ~2 experts. Negative",
         "control: E16C8 with free routing, where each task has one surviving expert. Same seeds (paired). A useful",
         "metric must move in the expected direction with a CI excluding 0; *standardised effect* = mean paired",
         "difference ÷ SD of the differences.\n",
         "| metric | expected (redundant − free) | redundant | free | paired diff [95% CI] | standardised effect | passes |",
         "|---|---|---|---|---|---|---|"]
    for label, fn, sign in metrics:
        a, b = seedmap(red, fn), seedmap(free, fn)
        a = {s: v for s, v in a.items() if np.isfinite(v)}
        b = {s: v for s, v in b.items() if np.isfinite(v)}
        c = compare(a, b)
        common = sorted(set(a) & set(b))
        d = np.array([a[s] - b[s] for s in common])
        eff = d.mean() / d.std(ddof=1) if len(d) > 1 and d.std(ddof=1) > 0 else float("nan")
        ok = (c["ci_lo"] > 0) if sign == "+" else (c["ci_hi"] < 0)
        L.append(f"| {label} | {sign} | {fmt(*mean_se(list(a.values())), 3)} | {fmt(*mean_se(list(b.values())), 3)} | "
                 f"{ci_str(c, 3)} | {eff:+.2f} | {'yes' if ok else 'no'} |")
    return L


def _sensitivity_plot(tuned):
    panels = [("aux", "aux loss α", lambda t: float(t[1:])), ("auxfree_bias", "bias update rate", lambda t: float(t[1:])),
              ("explore_noise", "exploration noise σ", lambda t: float(t[1:])), ("cosine_router", "cosine scale", lambda t: float(t[1:]))]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.3))
    for ax, (m, label, parse) in zip(axes, panels):
        if m not in tuned:
            continue
        grid = tuned[m]["grid_val_loss"]
        xs = sorted(grid, key=parse)
        for j, reg in enumerate(TUNE_REGIMES):
            ys = [grid[t][reg] for t in xs]
            ax.plot([parse(t) for t in xs], ys, "o-", color=SERIES[j], ms=4, lw=1.6, label=reg)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(label)
        ax.set_title(f"{m}: val loss vs setting", loc="left")
    axes[0].set_ylabel("val loss (tuning seeds)")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "e4_sensitivity.png", dpi=160)
    plt.close(fig)

    if "aux_then_release" in tuned:
        grid = tuned["aux_then_release"]["grid_val_loss"]
        alphas = sorted({float(t.split("_")[0][1:]) for t in grid})
        ts = sorted({int(t.split("_t")[1]) for t in grid})
        fig, axes = plt.subplots(1, 4, figsize=(14, 2.8))
        for ax, reg in zip(axes, TUNE_REGIMES):
            mat = np.array([[grid[f"a{a:g}_t{t}"][reg] for t in ts] for a in alphas])
            rel = mat / mat.min()
            im = ax.imshow(rel, cmap=SEQ.reversed(), vmin=1, vmax=max(1.5, rel.max()), aspect="auto")
            for i in range(len(alphas)):
                for j in range(len(ts)):
                    ax.text(j, i, f"{rel[i, j]:.2f}", ha="center", va="center", fontsize=7,
                            color="#fcfcfb" if rel[i, j] < 1.15 else "#0b0b0b")
            ax.set_xticks(range(len(ts)), ts)
            ax.set_yticks(range(len(alphas)), [f"{a:g}" for a in alphas])
            ax.set_xlabel("release step")
            ax.set_title(reg, loc="left")
            ax.grid(False)
        axes[0].set_ylabel("aux α while balancing")
        fig.suptitle("Balance-then-release: val loss relative to the best cell (1.00 = best)", x=0.01, ha="left", fontweight="bold")
        fig.tight_layout()
        fig.savefig(FIG / "e4_release_grid.png", dpi=160)
        plt.close(fig)


# ===================================================================================================== E5
E5_REGIMES = ["E16C16k2", "E32C16", "E8C24", "E16C16zipf08", "XOR8deep2", "CE16", "deep3E16C16"]
CONTINUOUS_BALANCE = ["aux", "auxfree_bias", "sinkhorn", "expert_choice"]


def analyze_e5():
    g = group(load_results("e5_heldout"))
    if not g:
        return
    test = lambda name: [V(r, "test_loss") for r in g.get(name, [])]  # noqa: E731
    mean = lambda name: float(np.mean(test(name))) if g.get(name) else float("nan")  # noqa: E731
    regs = [r for r in E5_REGIMES if g.get(f"{r}__none")]
    L = ["# E5 – Pre-registered held-out test\n",
         "Pre-registration: `experiments/PREREGISTRATION.md` (committed before any E5 run). Global configs from E4,",
         "10 fresh seeds, test split. Ratio = test loss ÷ best non-oracle, non-dense method in that regime.\n",
         "| method | " + " | ".join(regs) + " |", "|" + "---|" * (len(regs) + 1)]
    for m in METHODS + ["oracle", "dense_active", "dense_total"]:
        L.append(f"| {m} | " + " | ".join(fmt(*mean_se(test(f'{r}__{m}'))) for r in regs) + " |")
    best = {r: min(mean(f"{r}__{m}") for m in METHODS if g.get(f"{r}__{m}")) for r in regs}
    rows = []
    for m in METHODS:
        if not all(g.get(f"{r}__{m}") for r in regs):
            continue
        ratios = [mean(f"{r}__{m}") / best[r] for r in regs]
        gaps = [(mean(f"{r}__dense_active") - mean(f"{r}__{m}")) / (mean(f"{r}__dense_active") - mean(f"{r}__oracle")) for r in regs]
        rows.append((float(np.exp(np.mean(np.log(ratios)))), max(ratios), m, ratios, gaps))
    rows.sort()
    L += ["\n## Ratios and robustness\n", "| method | " + " | ".join(regs) + " | geo-mean | worst |", "|" + "---|" * (len(regs) + 3)]
    for gm, worst, m, ratios, _ in rows:
        L.append(f"| {m} | " + " | ".join(f"{x:.2f}" for x in ratios) + f" | {gm:.2f} | {worst:.2f} |")
    L += ["\n## Gap closed (0 = dense with active params, 1 = oracle routing)\n", "| method | " + " | ".join(regs) + " |",
          "|" + "---|" * (len(regs) + 1)]
    for _, _, m, _, gaps in rows:
        L.append(f"| {m} | " + " | ".join(f"{x:.2f}" for x in gaps) + " |")

    # ---- pre-registered predictions, evaluated mechanically
    geo = {m: gm for gm, _, m, _, _ in rows}
    worst = {m: w for _, w, m, _, _ in rows}
    P = []
    if "aux_then_release" in geo:
        leader = min(geo, key=geo.get)
        P.append(("P1", "aux_then_release has the lowest geo-mean ratio of the 13 methods",
                  f"lowest: {leader} ({geo[leader]:.3f}); aux_then_release {geo['aux_then_release']:.3f}", leader == "aux_then_release"))
        P.append(("P2", "aux_then_release's worst-case ratio ≤ 1.30", f"{worst['aux_then_release']:.3f}", worst["aux_then_release"] <= 1.30))
        wins = sum(mean(f"{r}__aux_then_release") < mean(f"{r}__none") for r in regs)
        P.append(("P3", "aux_then_release beats none in ≥ 6 of 7 regimes", f"{wins} of {len(regs)}", wins >= 6))
    for reg in ("E32C16", "E16C16zipf08"):
        if g.get(f"{reg}__none"):
            worse = [m for m in CONTINUOUS_BALANCE if mean(f"{reg}__{m}") > mean(f"{reg}__none")]
            P.append(("P4", f"in {reg}, ≥ 3 of 4 continuous-balance methods are worse than none",
                      f"worse than none: {', '.join(worse) or 'none of them'}", len(worse) >= 3))
    for reg, E, C in (("E32C16", 32, 16), ("E8C24", 8, 24), ("CE16", 16, 16)):
        if g.get(f"{reg}__none"):
            obs = float(np.mean([V(r, "dead_experts") for r in g[f"{reg}__none"]]))
            pred = random_owner_prediction(E, C)
            ok = abs(obs - pred) <= max(0.25 * pred, 1.0)
            P.append(("P5", f"{reg}: dead experts (none) within max(25%, 1) of random-owner prediction {pred:.2f}",
                      f"observed {obs:.2f}", ok))
    L += ["\n## Pre-registered predictions\n", "| id | prediction | result | verdict |", "|---|---|---|---|"]
    L += [f"| {pid} | {text} | {res} | {'**PASS**' if ok else '**FAIL**'} |" for pid, text, res, ok in P]
    (OUT / "summary_e5.md").write_text("\n".join(L) + "\n")


# ===================================================================================================== E6
E6_FAMILIES = ["clu_d1", "clu_d2", "clu_d4", "xor_d2", "xor_d4", "xor_d2_nopre"]
E6_CONDS = ["none", "frozen_router", "detach_input", "frozen_and_detached", "cosine_router", "aux", "auxfree_bias",
            "aux_then_release", "oracle"]


def _layers_final(r, key):
    f = r["final"]
    return [l[key] for l in f["layers"]] if "layers" in f else [f[key]]


def _dead0(r):
    h = r["history"][0]
    return float(np.mean([l["dead_experts"] for l in h["layers"]])) if "layers" in h else h["dead_experts"]


def _decomp_final(r, key):
    d = r["history"][-1].get("decomp")
    return float(np.mean([x[key] for x in d])) if d else float("nan")


def analyze_e6():
    g = group(load_results("e6_deep"))
    if not g:
        return
    L = ["# E6 – Depth and the representation pathway\n",
         "Dead experts are per MoE layer, averaged over layers. *rep share* = share of the routing-relevant change in",
         "router logits since step 0 that comes from the representations moving (vs the router weights), averaged over",
         "layers. *dead if only reps moved / only router moved* = dead experts under top-k of the counterfactual logits",
         "f(W₀, x_t) and f(W_t, x₀).\n"]
    preds = []
    for fam in E6_FAMILIES:
        if not g.get(f"{fam}__none"):
            continue
        base = g[f"{fam}__none"]
        L += [f"\n## {fam}\n",
              "| condition | dead / layer | dead at step 0 | Δ dead vs none [95% CI] | test loss | task info (last layer) | rep share | dead if only reps moved | dead if only router moved |",
              "|---|---|---|---|---|---|---|---|---|"]
        bd = seedmap(base, lambda r: float(np.mean(_layers_final(r, "dead_experts"))))
        for c in E6_CONDS:
            runs = g.get(f"{fam}__{c}")
            if not runs:
                continue
            dm = seedmap(runs, lambda r: float(np.mean(_layers_final(r, "dead_experts"))))
            cmp_ = compare(dm, bd) if c != "none" else None
            last_info = [(_layers_final(r, "label_info_recovered")[-1]) for r in runs]
            L.append(f"| {c} | {fmt(*mean_se(list(dm.values())), 2)} | {fmt(*mean_se([_dead0(r) for r in runs]), 2)} | "
                     f"{ci_str(cmp_) if cmp_ else '–'} | {fmt(*mean_se([V(r, 'test_loss') for r in runs]))} | "
                     f"{fmt(*mean_se(last_info), 2)} | {fmt(*mean_se([_decomp_final(r, 'rep_share') for r in runs]), 2)} | "
                     f"{fmt(*mean_se([_decomp_final(r, 'dead_rep_only') for r in runs]), 2)} | "
                     f"{fmt(*mean_se([_decomp_final(r, 'dead_router_only') for r in runs]), 2)} |")
        if fam in ("clu_d2", "clu_d4") and g.get(f"{fam}__frozen_router"):
            fr = g[f"{fam}__frozen_router"]
            c = compare(seedmap(fr, lambda r: float(np.mean(_layers_final(r, "dead_experts")))), seedmap(fr, _dead0))
            preds.append(("D1", f"{fam}: frozen router still creates dead experts beyond step 0", ci_str(c), c["ci_lo"] > 0))
        if fam in ("clu_d2", "clu_d4") and g.get(f"{fam}__detach_input"):
            c = compare(seedmap(g[f"{fam}__detach_input"], lambda r: float(np.mean(_layers_final(r, "dead_experts")))), bd)
            preds.append(("D2", f"{fam}: detaching the router input reduces dead experts vs none", ci_str(c), c["ci_hi"] < 0))
    if g.get("xor_d2_nopre__none"):
        info = float(np.mean([_layers_final(r, "label_info_recovered")[-1] for r in g["xor_d2_nopre__none"]]))
        preds.append(("D3", "xor without a dense pre-block: learned routing recovers < 0.3 of task information", f"{info:.2f}", info < 0.3))
    if preds:
        L += ["\n## Pre-registered predictions (E6)\n", "| id | prediction | result | verdict |", "|---|---|---|---|"]
        L += [f"| {pid} | {text} | {res} | {'**PASS**' if ok else '**FAIL**'} |" for pid, text, res, ok in preds]
    _e6_figure(g)
    (OUT / "summary_e6.md").write_text("\n".join(L) + "\n")


def _e6_figure(g):
    fams = [f for f in ("clu_d4", "xor_d4") if g.get(f"{f}__none")]
    if not fams:
        return
    fig, axes = plt.subplots(1, len(fams), figsize=(6.2 * len(fams), 3.6), squeeze=False)
    for ax, fam in zip(axes[0], fams):
        for j, c in enumerate(["none", "frozen_router", "detach_input", "frozen_and_detached"]):
            runs = g.get(f"{fam}__{c}")
            if not runs:
                continue
            steps = [h["step"] for h in runs[0]["history"]]
            y = np.mean([[np.mean([l["dead_experts"] for l in h["layers"]]) for h in r["history"]] for r in runs], axis=0)
            ax.plot(steps, y, color=SERIES[j], lw=2, label=c)
        ax.set_xscale("symlog", linthresh=10)
        ax.set_xlabel("training step")
        ax.set_ylabel("dead experts per layer")
        ax.set_title(fam, loc="left")
        ax.legend(fontsize=7)
    fig.suptitle("Deep models: which pathways create dead experts?", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "e6_pathways.png", dpi=160)
    plt.close(fig)


# ===================================================================================================== E3 v2
E3V2_ORDER = ["dense_active", "dense_total", "none", "aux_3e-3", "aux_1e-2", "aux_3e-2", "aux_1e-2_then_release",
              "auxfree_bias_1e-2", "bias_then_freeze", "hash_token", "frozen_router", "detach_input", "frozen_and_detached"]


def _lm_death_steps(r, total_steps):
    """Per layer: steps at which experts that are dead at the end died; plus experts that died and came back."""
    hist = [h for h in r["history"]]
    steps = [h["step"] for h in hist]
    out, revived = [], 0
    for li in range(len(hist[0]["layers"])):
        load = np.array([h["layers"][li]["load"] for h in hist])
        E = load.shape[1]
        for e in range(E):
            alive = load[:, e] >= 0.1 / E
            if not alive[-1]:
                idx = np.nonzero(alive)[0]
                out.append(steps[idx[-1] + 1] if len(idx) else 0)
            elif (~alive).any():
                revived += 1
    return out, revived


def analyze_e3v2(exp="e3v2_lm_cpu"):
    g = group(load_results(exp))
    if not g:
        return
    any_run = next(iter(g.values()))[0]
    total = any_run["config"]["steps"]
    doms = list(any_run["final"]["val_loss_by_domain"])
    L = [f"# E3 v2 – Scaled byte-level MoE LM ({exp})\n",
         f"{any_run['config']['n_layers']} layers, d_model {any_run['config']['d_model']}, {any_run['moe_config']['n_experts']} experts × "
         f"{any_run['moe_config']['d_hidden']} (top-{any_run['moe_config']['top_k']}), {total} steps. Mean ± s.e. over seeds;",
         "routing stats averaged over MoE layers. *late deaths* = experts dead at the end whose last alive evaluation was",
         f"after 10% of training (step {total // 10}). *revived* = experts that were dead at some evaluation and alive at the end.\n",
         "| method | val loss | " + " | ".join(doms) + " | effective experts | dead / layer | useful experts | domain info | byte-explained | late deaths / run | revived / run | n |",
         "|" + "---|" * (11 + len(doms))]
    from moe_lab.metrics import entropy_bits

    for m in E3V2_ORDER:
        runs = g.get(m)
        if not runs:
            continue
        vl = fmt(*mean_se([r["final"]["val_loss"] for r in runs]))
        pdm = [fmt(*mean_se([r["final"]["val_loss_by_domain"][d] for r in runs])) for d in doms]
        if runs[0]["moe_config"]["n_experts"] == 1:
            L.append(f"| {m} | {vl} | " + " | ".join(pdm) + f" | – | – | – | – | – | – | – | {len(runs)} |")
            continue
        Lm = lambda k: mean_se([np.mean([l.get(k, np.nan) for l in r["final"]["layers"]]) for r in runs])  # noqa: E731
        useful = mean_se([np.mean([2 ** entropy_bits(np.clip(a["ablation_delta"], 0, None)) for a in r["ablation"]]) for r in runs])
        late, rev = [], []
        for r in runs:
            ds, rv = _lm_death_steps(r, total)
            late.append(sum(d > total // 10 for d in ds))
            rev.append(rv)
        L.append(f"| {m} | {vl} | " + " | ".join(pdm) + f" | {fmt(*Lm('effective_experts'), 2)} | {fmt(*Lm('dead_experts'), 2)} | "
                 f"{fmt(*useful, 2)} | {fmt(*Lm('domain_info_recovered'), 2)} | {fmt(*Lm('routing_explained_by_byte'), 2)} | "
                 f"{np.mean(late):.2f} | {np.mean(rev):.2f} | {len(runs)} |")

    L += ["\n## Which pathway moves the router logits? (none / frozen router / detached input, final checkpoint)\n",
          "| method | rms change from reps | rms change from router weights | rep share | dead if only reps moved | dead if only router moved | dead (actual, probe) |",
          "|---|---|---|---|---|---|---|"]
    for m in ["none", "aux_1e-2", "frozen_router", "detach_input", "frozen_and_detached"]:
        runs = g.get(m)
        if not runs:
            continue
        D = lambda k: mean_se([_decomp_final(r, k) for r in runs])  # noqa: E731
        L.append(f"| {m} | {fmt(*D('rms_rep'), 2)} | {fmt(*D('rms_router'), 2)} | {fmt(*D('rep_share'), 2)} | "
                 f"{fmt(*D('dead_rep_only'), 2)} | {fmt(*D('dead_router_only'), 2)} | {fmt(*D('dead_both'), 2)} |")
    _e3v2_figure(g, total, exp)
    (OUT / f"summary_{exp.replace('_lm', '')}.md").write_text("\n".join(L) + "\n")


def _e3v2_figure(g, total, exp):
    show = [m for m in ["none", "aux_1e-2", "aux_1e-2_then_release", "frozen_router", "frozen_and_detached"] if g.get(m)]
    if not show:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 3.8))
    for j, m in enumerate(show):
        runs = g[m]
        steps = [h["step"] for h in runs[0]["history"]]
        eff = np.mean([[np.mean([l["effective_experts"] for l in h["layers"]]) for h in r["history"]] for r in runs], axis=0)
        axes[0].plot(steps, eff, color=SERIES[j], lw=2, label=m)
        rep = np.mean([[np.mean([d["rms_rep"] for d in h["decomp"]]) for h in r["history"]] for r in runs], axis=0)
        rout = np.mean([[np.mean([d["rms_router"] for d in h["decomp"]]) for h in r["history"]] for r in runs], axis=0)
        if m == "none":
            axes[1].plot(steps, rep, color=SERIES[0], lw=2, label="representations moved (router frozen at W₀)")
            axes[1].plot(steps, rout, color=SERIES[1], lw=2, label="router weights moved (inputs frozen at x₀)")
    axes[0].axvline(total // 10, color=INK2, lw=1, ls="--")
    axes[0].set_xscale("symlog", linthresh=25)
    axes[0].set_ylim(0, None)
    axes[0].set_xlabel("training step")
    axes[0].set_ylabel("effective experts (mean over layers)")
    axes[0].set_title("Load balance over training (dashed: release step)", loc="left")
    axes[0].legend(fontsize=7)
    axes[1].set_xscale("symlog", linthresh=25)
    axes[1].set_xlabel("training step")
    axes[1].set_ylabel("rms routing-relevant logit change")
    axes[1].set_title("No balancing: what moved the router logits?", loc="left")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / f"{exp.replace('_lm', '')}_dynamics.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    analyze_e1v2()
    analyze_e4()
    analyze_e5()
    analyze_e6()
    analyze_e3v2("e3v2_lm_cpu")
    analyze_e3v2("e3v2_lm_gpu")
    print("wrote", sorted(p.name for p in OUT.glob("summary_*.md")))
