"""Turn raw run JSONs into figures (results/figures/*.png) and summary tables (results/summary_*.md)."""
from __future__ import annotations

import math
import os
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from common import RESULTS, load_results  # noqa: E402

OUT = Path(os.environ.get("MOE_LAB_OUT", RESULTS))  # where figures/tables go (raw runs are always read from results/)
FIG = OUT / "figures"
# Reference categorical palette (fixed order, never cycled) + recessive ink.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SEQ = matplotlib.colors.LinearSegmentedColormap.from_list("seq", ["#f4f8fd", "#86b6ef", "#2a78d6", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "legend.frameon": False, "lines.linewidth": 2,
})


def group(results):
    g = defaultdict(list)
    for r in results:
        g[r["config"]["name"]].append(r)
    return g


def mean_se(vals):
    v = np.array([x for x in vals if x is not None and not (isinstance(x, float) and math.isnan(x))], dtype=float)
    if len(v) == 0:
        return float("nan"), float("nan")
    return float(v.mean()), float(v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else 0.0


def fmt(m, s, digits=3):
    return "–" if math.isnan(m) else f"{m:.{digits}f} ± {s:.{digits}f}"


def dense_equivalent(groups, regime, expert_width=8):
    """Map a val loss to the width of a dense MLP reaching the same loss (log-linear interpolation)."""
    pts = []
    for name, runs in groups.items():
        if name.startswith(f"{regime}__dense_"):
            w = int(name.rsplit("_", 1)[1])
            pts.append((w, np.mean([r["final"]["val_loss"] for r in runs])))
    pts.sort()
    if len(pts) < 2:
        return lambda loss: float("nan")
    ws = np.log2([p[0] for p in pts])
    ls = np.array([p[1] for p in pts])
    # Dense loss is (nearly) monotone in width; enforce it so interpolation is well defined.
    ls = np.minimum.accumulate(ls)

    def f(loss):
        if loss <= ls[-1]:
            return float("inf")
        if loss >= ls[0]:
            return 2 ** ws[0] / expert_width
        i = np.searchsorted(-ls, -loss)
        t = (loss - ls[i - 1]) / (ls[i] - ls[i - 1])
        return 2 ** (ws[i - 1] + t * (ws[i] - ws[i - 1])) / expert_width

    return f


# ------------------------------------------------------------------------------------- E1
E1_ORDER = ["baseline", "frozen_router", "frozen_experts", "identical_init", "zero_init_output", "router_warmup_100",
            "router_lr_x0.1", "router_lr_x10", "adam_load_normalized", "sgd", "sgd_load_normalized", "head_start_200",
            "topk_renorm_k1"]


def experts_per_task(r):
    """exp(entropy) of how each latent task's tokens are spread over experts, averaged over tasks (1 = one owner)."""
    from moe_lab.metrics import entropy_bits

    cte = np.array(r["final"]["cluster_to_expert"])
    return float(np.mean([2 ** entropy_bits(row) for row in cte if row.sum() > 0]))


def death_steps(r):
    """For each expert dead at the end: the step after which its load never again exceeded 10% of fair share."""
    E = len(r["final"]["load"])
    steps = [h["step"] for h in r["history"]]
    load = np.array([h["load"] for h in r["history"]])
    out = []
    for e in range(E):
        alive = load[:, e] >= 0.1 / E
        if alive[-1]:
            continue
        idx = np.nonzero(alive)[0]
        out.append(steps[idx[-1] + 1] if len(idx) else 0)
    return out


def random_owner_prediction(E, C):
    """Expected #experts owning no task if each task's owner were an independent uniform draw."""
    return E * (1 - 1 / E) ** C


def analyze_e1():
    res = load_results("e1_mechanisms")
    if not res:
        return
    g = group(res)
    lines = ["# E1 – Mechanisms of collapse (synthetic)\n",
             "Mean ± standard error over seeds. `dead` = experts receiving <10% of a fair share of tokens.\n"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.36 * len(E1_ORDER) + 1.4), sharey=True)
    for ax, regime in zip(axes, ["E8", "E32"]):
        E = 8 if regime == "E8" else 32
        lines += [f"\n## Regime {regime} ({E} experts, {E} latent tasks)\n",
                  "| condition | dead experts | dead at step 0 | effective experts | experts per task | task info recovered | val loss | n |",
                  "|---|---|---|---|---|---|---|---|"]
        names, means, ses = [], [], []
        for cond in E1_ORDER:
            runs = g.get(f"{regime}__{cond}", [])
            if not runs:
                continue
            d = mean_se([r["final"]["dead_experts"] for r in runs])
            d0 = mean_se([r["history"][0]["dead_experts"] for r in runs])
            ee = mean_se([r["final"]["effective_experts"] for r in runs])
            rec = mean_se([r["final"]["label_info_recovered"] for r in runs])
            vl = mean_se([r["final"]["val_loss"] for r in runs])
            ept = mean_se([experts_per_task(r) for r in runs])
            lines.append(f"| {cond} | {fmt(*d, 2)} | {fmt(*d0, 2)} | {fmt(*ee, 2)} | {fmt(*ept, 2)} | {fmt(*rec, 2)} | {fmt(*vl)} | {len(runs)} |")
            names.append(cond)
            means.append(d[0] / E)
            ses.append(d[1] / E)
        y = np.arange(len(names))[::-1]
        ax.barh(y, means, xerr=ses, color=SERIES[0], height=0.6, error_kw=dict(ecolor=INK2, lw=1, capsize=2))
        ax.set_yticks(y, names)
        ax.set_xlabel("fraction of experts dead at end of training")
        ax.set_title(f"{regime}: {E} experts, {E} latent tasks", loc="left")
        ax.grid(axis="y", visible=False)
        pad = 0.012 * max(means)
        for yi, m, se in zip(y, means, ses):
            ax.text(m + se + 2 * pad, yi, f"{m:.2f}", va="center", ha="left", color=INK2, fontsize=8)
        init = np.mean([r["history"][0]["dead_experts"] for r in g[f"{regime}__baseline"]]) / E
        if init > 0:
            ax.axvline(init, color=INK2, lw=1, ls="--")
            ax.text(init, y[0] + 0.55, " dead already at init", color=INK2, fontsize=7, va="bottom")
        ax.set_xlim(0, max(m + s for m, s in zip(means, ses)) * 1.15)
    fig.suptitle("Which link of the feedback loop causes dead experts? (no load balancing in any condition)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "e1_mechanisms_dead_experts.png", dpi=160)
    plt.close(fig)

    lines += ["\n## When do experts die, and how many would a random owner per task predict?\n",
              "| regime | dead (observed) | random-owner prediction E·(1−1/E)^C | experts per task | median death step | deaths before step 100 | n |",
              "|---|---|---|---|---|---|---|"]
    for regime, E, C in (("E8", 8, 8), ("E16C8", 16, 8), ("E32", 32, 32)):
        runs = g.get(f"{regime}__baseline", [])
        if not runs:
            continue
        ds = [x for r in runs for x in death_steps(r)]
        d = mean_se([r["final"]["dead_experts"] for r in runs])
        ept = mean_se([experts_per_task(r) for r in runs])
        med = f"{np.median(ds):.0f}" if ds else "–"
        early = f"{np.mean(np.array(ds) < 100):.0%}" if ds else "–"
        lines.append(f"| {regime} | {fmt(*d, 2)} | {random_owner_prediction(E, C):.2f} | {fmt(*ept, 2)} | {med} | {early} | {len(runs)} |")
    hs = g.get("E8__head_start_200", [])
    if hs:
        share0 = mean_se([r["final"]["load"][0] for r in hs])
        lines.append(f"\nHead start: expert 0 (trained alone for 200 steps first) ends with {fmt(*share0, 2)} of tokens "
                     f"(fair share {1 / 8:.3f}).")
    _plot_dynamics(g)
    _overfitting(g, lines)
    (OUT / "summary_e1.md").write_text("\n".join(lines) + "\n")


def _plot_dynamics(g):
    runs = g.get("E8__baseline", [])
    if not runs:
        return
    r = max(runs, key=lambda r: (r["final"]["dead_experts"], -r["config"]["seed"]))
    steps = [h["step"] for h in r["history"]]
    load = np.array([h["load"] for h in r["history"]])
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    ax = axes[0]
    for e in range(load.shape[1]):
        ax.plot(steps, load[:, e], color=SERIES[e], lw=1.8, label=f"expert {e}")
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xlabel("training step")
    ax.set_ylabel("share of tokens")
    ax.set_title(f"Token share per expert (seed {r['config']['seed']})", loc="left")
    ax.legend(ncol=2, fontsize=7, loc="upper right")
    ax = axes[1]
    ax.plot(steps, [h["router_entropy_norm"] for h in r["history"]], color=SERIES[0], label="router entropy (norm.)")
    ax.plot(steps, [h["label_info_recovered"] for h in r["history"]], color=SERIES[1], label="task info in routing")
    ax.set_xscale("symlog", linthresh=10)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("training step")
    ax.set_title("Router becomes confident early", loc="left")
    ax.legend(fontsize=7)
    ax = axes[2]
    ax.plot(steps, [h["router_logit_std"] for h in r["history"]], color=SERIES[0])
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xlabel("training step")
    ax.set_title("Router logit scale (std across experts)", loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "e1_dynamics.png", dpi=160)
    plt.close(fig)

    snaps = r["snapshots"]
    keys = sorted(snaps, key=int)
    fig, axes = plt.subplots(1, len(keys) + 1, figsize=(3.2 * (len(keys) + 1), 3.3))
    mats = [np.array(snaps[k]["expert_cluster_loss"]) for k in keys] + [None]
    vmax = max(m.max() for m in mats if m is not None)
    for ax, k, m in zip(axes[:-1], keys, mats):
        cte = np.array(snaps[k]["cluster_to_expert"])  # [C, E]
        im = ax.imshow(m, cmap=SEQ.reversed(), vmin=0, vmax=vmax, aspect="auto")
        owner = cte.argmax(1)
        for c, e in enumerate(owner):
            ax.add_patch(plt.Rectangle((c - 0.5, e - 0.5), 1, 1, fill=False, ec=SERIES[1], lw=1.6))
        ax.set_title(f"step {k}", loc="left")
        ax.set_xlabel("latent task (cluster)")
        ax.set_ylabel("expert")
        ax.grid(False)
    fin = np.array(r["final"]["expert_cluster_loss"])
    ax = axes[-1]
    im = ax.imshow(fin, cmap=SEQ.reversed(), vmin=0, vmax=vmax, aspect="auto")
    for c, e in enumerate(np.array(r["final"]["cluster_to_expert"]).argmax(1)):
        ax.add_patch(plt.Rectangle((c - 0.5, e - 0.5), 1, 1, fill=False, ec=SERIES[1], lw=1.6))
    ax.set_title(f"step {r['config']['steps']} (final)", loc="left")
    ax.set_xlabel("latent task (cluster)")
    ax.grid(False)
    fig.colorbar(im, ax=axes, shrink=0.8, label="loss if this expert handled the task (darker = better)")
    fig.suptitle("Counterfactual loss of every expert on every task; orange box = expert the router actually uses",
                 x=0.01, ha="left", fontweight="bold")
    fig.savefig(FIG / "e1_expert_task_loss.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _overfitting(g, lines):
    lines += ["\n## Is a popular expert *overfit*?\n",
              "Per-expert loss on the tokens routed to it. If popular experts were overfitting, their train/val",
              "gap would grow with load. Spearman correlations are computed per run, then averaged.\n",
              "| regime | ρ(load, train loss) | ρ(load, val−train gap) | mean gap, experts serving 1 task | mean gap, experts serving ≥2 tasks | n |",
              "|---|---|---|---|---|---|"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for regime, color in (("E8__baseline", SERIES[0]), ("overfit__baseline", SERIES[1])):
        runs = g.get(regime, [])
        if not runs:
            continue
        rho_tr, rho_gap, gap1, gap2, xs, ytr, ygap = [], [], [], [], [], [], []
        for r in runs:
            f = r["final"]
            n_tasks = (np.array(f["cluster_to_expert"]).argmax(1)[:, None] == np.arange(len(f["load"]))).sum(0)
            tok = np.array(f["per_expert_train_tokens"], float)
            tr = np.array(f["per_expert_train_loss"], float)
            gap = np.array(f["per_expert_gap"], float)
            ok = ~np.isnan(tr) & ~np.isnan(gap)
            if ok.sum() >= 3:
                from moe_lab.metrics import spearman

                rho_tr.append(spearman(tok[ok], tr[ok]))
                rho_gap.append(spearman(tok[ok], gap[ok]))
            gap1 += gap[ok & (n_tasks == 1)].tolist()
            gap2 += gap[ok & (n_tasks >= 2)].tolist()
            xs += (tok[ok] / tok.sum()).tolist()
            ytr += tr[ok].tolist()
            ygap += gap[ok].tolist()
        lines.append(f"| {regime} | {fmt(*mean_se(rho_tr), 2)} | {fmt(*mean_se(rho_gap), 2)} | {fmt(*mean_se(gap1))} | "
                     f"{fmt(*mean_se(gap2))} | {len(runs)} |")
        lab = "plentiful data (1000/task)" if regime.startswith("E8") else "scarce data (100/task)"
        axes[0].scatter(xs, ytr, s=22, color=color, alpha=0.75, edgecolor="#fcfcfb", lw=0.6, label=lab)
        axes[1].scatter(xs, ygap, s=22, color=color, alpha=0.75, edgecolor="#fcfcfb", lw=0.6, label=lab)
    axes[0].set_xlabel("share of training tokens routed to the expert")
    axes[0].set_ylabel("train loss on its tokens")
    axes[0].set_title("Busy experts fit their tokens *worse* (underfit)", loc="left")
    axes[1].set_xlabel("share of training tokens routed to the expert")
    axes[1].set_ylabel("val loss − train loss on its tokens")
    axes[1].axhline(0, color=INK2, lw=1)
    axes[1].set_title("Generalisation gap vs. load", loc="left")
    for ax in axes:
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "e1_overfitting_test.png", dpi=160)
    plt.close(fig)


# ------------------------------------------------------------------------------------- E2
E2_ORDER = ["none", "aux_1e-3", "aux_1e-2", "aux_1e-1", "aux_1", "zloss_1e-3", "aux_1e-2+zloss", "explore_noise_1",
            "auxfree_bias_1e-2", "expert_choice", "sinkhorn", "cosine_router", "revive_dead", "aux_1e-1_then_release",
            "auxfree_bias_then_freeze", "sinkhorn_then_release", "hash_fixed", "oracle"]


def analyze_e2():
    res = load_results("e2_fixes")
    if not res:
        return
    g = group(res)
    lines = ["# E2 – Fixes (synthetic)\n",
             "Mean ± s.e. over seeds. *dense-equiv.* = width of a plain MLP reaching the same val loss, in units of one",
             "expert (8 = as good as a dense model with all experts' parameters active; 1 = no better than one expert).",
             "*task info* = MI(routing; true task)/H(task). *expert CKA* = mean pairwise similarity of alive experts' functions.\n"]
    for regime in ["E8", "E32", "E16C8", "E8zipf"]:
        de = dense_equivalent(g, regime)
        rows = []
        for m in E2_ORDER:
            runs = g.get(f"{regime}__{m}", [])
            if not runs:
                continue
            F = lambda k: mean_se([r["final"][k] for r in runs])  # noqa: E731
            vl = F("val_loss")
            row = dict(method=m, val=vl, eff=F("effective_experts"), dead=F("dead_experts"), rec=F("label_info_recovered"),
                       cka=F("alive_expert_cka_mean"), useful=F("useful_experts"), deq=de(vl[0]), n=len(runs),
                       ept=mean_se([experts_per_task(r) for r in runs]))
            if regime == "E8zipf":
                pcl = np.array([r["final"]["per_cluster_val_loss"] for r in runs])
                row["rare"] = mean_se(pcl[:, -3:].mean(1))  # three rarest tasks
                row["common"] = mean_se(pcl[:, 0])
            rows.append(row)
        if not rows:
            continue
        dense_rows = sorted(((int(n.rsplit("_", 1)[1]), runs) for n, runs in g.items() if n.startswith(f"{regime}__dense_")))
        lines += [f"\n## Regime {regime}\n"]
        hdr = "| method | val loss | dense-equiv. experts | effective experts | dead | experts per task | task info | useful experts (ablation) | expert CKA |"
        if regime == "E8zipf":
            hdr += " loss on most common task | loss on 3 rarest tasks |"
        lines += [hdr, "|" + "---|" * (hdr.count("|") - 1)]
        for row in rows:
            deq = "> all" if math.isinf(row["deq"]) else f"{row['deq']:.1f}"
            line = (f"| {row['method']} | {fmt(*row['val'])} | {deq} | {fmt(*row['eff'], 2)} | {fmt(*row['dead'], 2)} | {fmt(*row['ept'], 2)} | "
                    f"{fmt(*row['rec'], 2)} | {fmt(*row['useful'], 2)} | {fmt(*row['cka'], 2)} |")
            if regime == "E8zipf":
                line += f" {fmt(*row['common'])} | {fmt(*row['rare'])} |"
            lines.append(line)
        lines += ["", "Dense baselines: " + ", ".join(
            f"width {w}: {np.mean([r['final']['val_loss'] for r in runs]):.3f}" for w, runs in dense_rows)]
        _plot_fixes(regime, rows, dense_rows)
    (OUT / "summary_e2.md").write_text("\n".join(lines) + "\n")


def _plot_fixes(regime, rows, dense_rows):
    panels = [("val", "val loss (log scale, lower is better)"), ("eff", "effective experts (balance of load)"),
              ("rec", "task info recovered by routing"), ("useful", "useful experts (loss rises if ablated)")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 0.32 * len(rows) + 2.0), sharey=True)
    y = np.arange(len(rows))[::-1]
    for ax, (key, title) in zip(axes, panels):
        m = [r[key][0] for r in rows]
        s = [r[key][1] for r in rows]
        ax.errorbar(m, y, xerr=s, fmt="o", ms=5, color=SERIES[0], ecolor=INK2, elinewidth=1, capsize=2)
        ax.set_title(title, loc="left", fontsize=9)
        ax.grid(axis="y", visible=False)
        if key == "val" and dense_rows:
            for w, runs in dense_rows:
                if w in (8, 64, 256):
                    v = np.mean([r["final"]["val_loss"] for r in runs])
                    ax.axvline(v, color=SERIES[1], lw=1, ls="--")
                    ax.text(v, y[-1] - 0.9, f"dense\nwidth {w}", color=INK2, fontsize=6.5, va="top", ha="center")
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2g"))
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            lo = min(m) * 0.85
            ax.set_xticks([t for t in (0.1, 0.15, 0.2, 0.3, 0.5, 0.8) if t >= lo])
            ax.set_ylim(y[-1] - 2.2, y[0] + 0.7)
        if key in ("eff", "useful"):
            E = max(r["eff"][0] for r in rows)
            ax.axvline(round(E), color=INK2, lw=1, ls=":")
    axes[0].set_yticks(y, [r["method"] for r in rows])
    fig.suptitle(f"Regime {regime}: balancing methods compared (mean ± s.e. over seeds)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / f"e2_fixes_{regime}.png", dpi=160)
    plt.close(fig)


# ------------------------------------------------------------------------------------- E3
E3_ORDER = ["dense_active", "dense_total", "none", "aux_1e-2_batch", "aux_1e-2_sequence", "aux_1e-2+zloss",
            "auxfree_bias_1e-2", "hash_token", "frozen_router"]


def analyze_e3():
    res = load_results("e3_lm")
    if not res:
        return
    g = group(res)
    doms = list(res[0]["final"]["val_loss_by_domain"])
    lines = ["# E3 – Tiny byte-level MoE language model (3 domains)\n",
             "Val loss in nats/byte, mean ± s.e. over seeds. Per-layer routing stats are averaged over the 2 MoE layers.",
             "*domain info* = MI(top-1 expert; domain)/H(domain). *byte-explained* = MI(top-1 expert; current byte)/H(expert).\n",
             "| method | val loss | " + " | ".join(doms) + " | effective experts (of 8) | dead | domain info | byte-explained | expert CKA |",
             "|" + "---|" * (6 + len(doms))]
    for m in E3_ORDER:
        runs = g.get(m, [])
        if not runs:
            continue
        vl = mean_se([r["final"]["val_loss"] for r in runs])
        per_dom = [fmt(*mean_se([r["final"]["val_loss_by_domain"][d] for r in runs])) for d in doms]
        L = lambda k: mean_se([np.mean([l.get(k, np.nan) for l in r["final"]["layers"]]) for r in runs])  # noqa: E731
        if runs[0]["moe_config"]["n_experts"] == 1:
            rest = "– | – | – | – | –"
        else:
            rest = (f"{fmt(*L('effective_experts'), 2)} | {fmt(*L('dead_experts'), 1)} | {fmt(*L('domain_info_recovered'), 2)} | "
                    f"{fmt(*L('routing_explained_by_byte'), 2)} | {fmt(*L('alive_expert_cka_mean'), 2)}")
        lines.append(f"| {m} | {fmt(*vl)} | " + " | ".join(per_dom) + f" | {rest} |")
    (OUT / "summary_e3.md").write_text("\n".join(lines) + "\n")

    show = [m for m in ["none", "aux_1e-2_batch", "aux_1e-2_sequence", "auxfree_bias_1e-2"] if m in g]
    if not show:
        return
    n_layers = len(g[show[0]][0]["final"]["layers"])
    fig, axes = plt.subplots(n_layers, len(show), figsize=(3.4 * len(show), 2.6 * n_layers), squeeze=False)
    for j, m in enumerate(show):
        r = sorted(g[m], key=lambda r: r["config"]["seed"])[0]
        for li in range(n_layers):
            ax = axes[li, j]
            mat = np.array(r["final"]["layers"][li]["domain_expert_load"])  # [D, E]
            im = ax.imshow(mat, cmap=SEQ, vmin=0, vmax=max(0.5, mat.max()), aspect="auto")
            ax.set_yticks(range(len(doms)), doms if j == 0 else [""] * len(doms))
            ax.set_xticks(range(mat.shape[1]))
            ax.grid(False)
            if li == 0:
                ax.set_title(m, loc="left")
            if li == n_layers - 1:
                ax.set_xlabel("expert")
            if j == 0:
                ax.set_ylabel(f"layer {li}")
    fig.colorbar(im, ax=axes, shrink=0.7, label="share of the domain's tokens")
    fig.suptitle("Where each domain's tokens go (seed 0)", x=0.01, ha="left", fontweight="bold")
    fig.savefig(FIG / "e3_domain_routing.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.6))
    for j, m in enumerate([m for m in E3_ORDER if m in g]):
        r = sorted(g[m], key=lambda r: r["config"]["seed"])[0]
        steps = [h["step"] for h in r["history"]]
        ax.plot(steps, [h["val_loss"] for h in r["history"]], color=SERIES[j % 8] if j < 8 else INK2, lw=1.6, label=m)
    ax.set_xlabel("training step")
    ax.set_ylabel("val loss (nats/byte)")
    ax.set_ylim(top=2.6)
    ax.legend(fontsize=7, ncol=2)
    ax.set_title("Validation loss (seed 0)", loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "e3_val_loss.png", dpi=160)
    plt.close(fig)


def export_csv():
    """One row per run with every scalar final metric (raw JSONs are large and not committed)."""
    import csv

    for exp in ("e1_mechanisms", "e2_fixes", "e3_lm"):
        res = load_results(exp)
        if not res:
            continue
        rows = []
        for r in res:
            row = {"name": r["config"]["name"], "seed": r["config"]["seed"]}
            f = r["final"]
            if exp == "e3_lm":
                row["val_loss"] = f["val_loss"]
                row.update({f"val_loss_{k}": v for k, v in f["val_loss_by_domain"].items()})
                for li, layer in enumerate(f["layers"]):
                    row.update({f"L{li}_{k}": v for k, v in layer.items() if isinstance(v, (int, float))})
                    row[f"L{li}_load"] = " ".join(f"{x:.4f}" for x in layer["load"])
            else:
                row.update({k: v for k, v in f.items() if isinstance(v, (int, float))})
                row["load"] = " ".join(f"{x:.4f}" for x in f["load"])
                row["experts_per_task"] = experts_per_task(r)
            rows.append(row)
        keys = list(dict.fromkeys(k for row in rows for k in row))
        with open(OUT / f"runs_{exp}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    export_csv()
    analyze_e1()
    analyze_e2()
    analyze_e3()
    print("wrote", sorted(p.name for p in OUT.glob("summary_*.md")), sorted(p.name for p in FIG.glob("*.png")))
