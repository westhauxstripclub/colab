"""Paired statistics for seed-matched comparisons.

Every condition run with seed s shares its task and (where the architecture matches) its initialisation with every
other condition run with seed s, so comparisons between conditions are paired. Paired tests remove the large
seed-to-seed variation (some tasks are simply easier) and are much more sensitive than comparing two means.
"""
from __future__ import annotations

import numpy as np


def paired(a: dict, b: dict) -> tuple[np.ndarray, np.ndarray]:
    """Align two {seed: value} maps on their common seeds."""
    seeds = sorted(set(a) & set(b))
    return np.array([a[s] for s in seeds], dtype=float), np.array([b[s] for s in seeds], dtype=float)


def bootstrap_ci(d: np.ndarray, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """Mean of d with a percentile bootstrap (1 - alpha) confidence interval."""
    d = np.asarray(d, dtype=float)
    if len(d) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(d.mean()), float(lo), float(hi)


def sign_flip_pvalue(d: np.ndarray, n_perm: int = 20_000, seed: int = 0) -> float:
    """Two-sided paired permutation test of mean(d) == 0 (randomly flips the sign of each paired difference)."""
    d = np.asarray(d, dtype=float)
    if len(d) == 0 or np.all(d == 0):
        return 1.0
    rng = np.random.default_rng(seed)
    obs = abs(d.mean())
    flips = rng.choice([-1.0, 1.0], size=(n_perm, len(d)))
    null = np.abs((flips * d).mean(1))
    return float((1 + (null >= obs - 1e-12).sum()) / (n_perm + 1))


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (controls the family-wise error rate across a table of comparisons)."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj.tolist()


def compare(treatment: dict, control: dict, **kw) -> dict:
    """Paired effect of treatment - control over shared seeds: mean, 95% CI, permutation p, n."""
    t, c = paired(treatment, control)
    d = t - c
    mean, lo, hi = bootstrap_ci(d, **kw)
    return {"diff": mean, "ci_lo": lo, "ci_hi": hi, "p": sign_flip_pvalue(d), "n": len(d)}
