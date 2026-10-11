"""Diagnostics that separate "balanced" from "actually a mixture of experts".

Load balance alone is not enough: a perfectly balanced router can still send tokens to interchangeable
experts (still a dense model in disguise). So we measure four different things:
  * load        - how tokens are spread (effective #experts, dead experts)
  * specialise  - mutual information between routing and ground-truth structure
  * redundancy  - how similar the experts' functions are (CKA on shared probe inputs)
  * utility     - how much loss each expert is responsible for (ablation) and how far routing is from the
                  best available expert (regret)
"""
from __future__ import annotations

import math

import numpy as np
import torch


def entropy_bits(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 0]
    p = p / p.sum()
    return float(-(p * np.log2(p)).sum())


def load_stats(M: torch.Tensor) -> dict:
    """M: [N, E] dispatch mask. Returns load distribution summaries."""
    counts = M.float().sum(0).cpu().numpy()
    E = counts.shape[0]
    frac = counts / max(counts.sum(), 1.0)
    H = entropy_bits(frac) if counts.sum() > 0 else 0.0
    return {
        "load": frac.tolist(),
        "effective_experts": float(2**H),  # exp(entropy): E at perfect balance, 1 at total collapse
        "max_over_mean": float(frac.max() * E),
        "dead_experts": int((frac < 0.1 / E).sum()),  # experts receiving <10% of their fair share
        "load_cv": float(frac.std() / max(frac.mean(), 1e-12)),
        "tokens_dropped_frac": float((M.sum(1) == 0).float().mean()),
    }


def router_stats(logits: torch.Tensor) -> dict:
    p = logits.softmax(-1)
    ent = -(p * p.clamp_min(1e-12).log()).sum(-1).mean().item()
    return {
        "router_entropy_norm": ent / math.log(logits.shape[-1]) if logits.shape[-1] > 1 else 0.0,  # 1 = uniform
        "router_logit_std": logits.std(-1).mean().item() if logits.shape[-1] > 1 else 0.0,
        "router_max_prob": p.max(-1).values.mean().item(),
    }


def mutual_info_bits(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """Discrete MI(a; b) and the marginal entropies, all in bits."""
    a = np.asarray(a).astype(np.int64)
    b = np.asarray(b).astype(np.int64)
    joint = np.zeros((a.max() + 1, b.max() + 1))
    np.add.at(joint, (a, b), 1)
    joint /= joint.sum()
    pa, pb = joint.sum(1), joint.sum(0)
    nz = joint > 0
    mi = float((joint[nz] * np.log2(joint[nz] / (pa[:, None] * pb[None, :])[nz])).sum())
    return mi, entropy_bits(pa), entropy_bits(pb)


def specialization(expert_top1: np.ndarray, labels: np.ndarray) -> dict:
    mi, h_e, h_y = mutual_info_bits(expert_top1, labels)
    return {
        "mi_bits": mi,
        "label_info_recovered": mi / h_y if h_y > 0 else 0.0,  # 1 = routing reveals the label exactly
        "routing_explained": mi / h_e if h_e > 0 else 0.0,  # share of routing entropy explained by label
    }


def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    xy = (Y.t() @ X).pow(2).sum()
    xx = (X.t() @ X).pow(2).sum().sqrt()
    yy = (Y.t() @ Y).pow(2).sum().sqrt()
    return float(xy / (xx * yy).clamp_min(1e-12))


def expert_redundancy(outputs: torch.Tensor, alive: np.ndarray | None = None) -> dict:
    """outputs: [P, E, d] expert outputs on a shared probe set. Mean pairwise CKA (1 = identical functions)."""
    E = outputs.shape[1]
    idx = list(range(E)) if alive is None else [e for e in range(E) if alive[e]]
    sims = [linear_cka(outputs[:, i], outputs[:, j]) for a, i in enumerate(idx) for j in idx[a + 1 :]]
    return {"expert_cka_mean": float(np.mean(sims)) if sims else float("nan"), "n_compared": len(idx)}


def gini(x: np.ndarray) -> float:
    x = np.sort(np.abs(np.asarray(x, dtype=np.float64)))
    if x.sum() == 0:
        return 0.0
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def spearman(a, b) -> float:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    ra, rb = a.argsort().argsort(), b.argsort().argsort()
    return float(np.corrcoef(ra, rb)[0, 1])


def routed_cka(outputs: torch.Tensor, top1: torch.Tensor, alive: np.ndarray, min_tokens: int = 20) -> float:
    """Mean pairwise CKA between experts, evaluated on the tokens actually routed to either expert of the pair.

    Plain CKA over a shared probe set mostly compares experts on inputs they never see; restricting to the union of
    their routed tokens asks "would these two experts compute the same thing on the data they are responsible for?".
    """
    idx = [e for e in range(outputs.shape[1]) if alive[e] and (top1 == e).sum() >= min_tokens]
    sims = []
    for a, i in enumerate(idx):
        for j in idx[a + 1 :]:
            sel = (top1 == i) | (top1 == j)
            sims.append(linear_cka(outputs[sel, i], outputs[sel, j]))
    return float(np.mean(sims)) if sims else float("nan")


def substitution_gap(cf_loss: torch.Tensor, top1: torch.Tensor, alive: np.ndarray, min_tokens: int = 20) -> float:
    """How much worse the best *other* expert is on an expert's own tokens, relative to that expert (mean over alive
    experts). Near 0 = the expert is replaceable (redundant); large = it does something nobody else can."""
    if cf_loss.shape[1] < 2:
        return float("nan")
    gaps = []
    for e in range(cf_loss.shape[1]):
        sel = top1 == e
        if not alive[e] or sel.sum() < min_tokens:
            continue
        own = cf_loss[sel, e].mean()
        others = torch.cat([cf_loss[sel, :e], cf_loss[sel, e + 1 :]], dim=1).min(1).values.mean()
        gaps.append(float((others - own) / own.clamp_min(1e-8)))
    return float(np.mean(gaps)) if gaps else float("nan")


def _centered_rms(t: torch.Tensor) -> float:
    """Size of a logit change that can affect routing: per-token shifts common to all experts are removed."""
    return float((t - t.mean(-1, keepdim=True)).pow(2).mean().sqrt())


def logit_decomposition(layer, x0: torch.Tensor, xt: torch.Tensor, params0: dict, k: int) -> dict:
    """Split the change in router logits since step 0 into a representation term and a router-weight term.

        total  = f(W_t, x_t) - f(W_0, x_0)
        rep    = f(W_0, x_t) - f(W_0, x_0)   (router frozen, inputs moved)
        router = f(W_t, x_0) - f(W_0, x_0)   (inputs frozen, router moved)
        interaction = total - rep - router

    Also reports how balanced routing would be if only one pathway had acted (effective experts / dead experts
    under top-k of the rep-only and router-only logits). Works for any router type via functional evaluation.
    """
    L00 = layer.router_logits_with(x0, params0)
    L0t = layer.router_logits_with(xt, params0)
    Lt0 = layer.router_logits_with(x0)
    Ltt = layer.router_logits_with(xt)
    rep, rout, total = L0t - L00, Lt0 - L00, Ltt - L00
    inter = total - rep - rout
    E = L00.shape[-1]

    def load_of(logits):
        M = torch.zeros_like(logits, dtype=torch.bool).scatter_(1, logits.topk(min(k, E), -1).indices, True)
        st = load_stats(M)
        return st["effective_experts"], st["dead_experts"]

    r_rep, r_rout, r_int = _centered_rms(rep), _centered_rms(rout), _centered_rms(inter)
    eff_rep, dead_rep = load_of(L0t)
    eff_rout, dead_rout = load_of(Lt0)
    eff_tot, dead_tot = load_of(Ltt)
    return {
        "rms_rep": r_rep,
        "rms_router": r_rout,
        "rms_interaction": r_int,
        "rms_total": _centered_rms(total),
        "rep_share": r_rep**2 / max(r_rep**2 + r_rout**2, 1e-12),
        "eff_rep_only": eff_rep,
        "dead_rep_only": dead_rep,
        "eff_router_only": eff_rout,
        "dead_router_only": dead_rout,
        "eff_both": eff_tot,
        "dead_both": dead_tot,
    }
