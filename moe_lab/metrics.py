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
