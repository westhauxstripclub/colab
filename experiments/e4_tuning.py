"""E4 - Per-method hyperparameter tuning, with selection kept separate from evaluation.

Stages (run in order):
  tune    every method's grid on the four round-1 regimes, *tuning seeds* 100-103; scored on the validation split
  select  pick per method (a) the best config per regime and (b) one global config (lowest geometric-mean ratio to
          the best config in each regime), and write results/tuned_configs.json
  eval    run the selected configs on fresh *evaluation seeds* 0-9 and report the held-out *test* split, plus oracle
          routing, a dense-width sweep (for "gap closed" and dense-equivalent size) and the redundancy-metric controls

Selecting and reporting on the same seeds/split would reward luck; the global config is what a practitioner who
tunes once and deploys everywhere would get, so it is the fair basis for robustness claims (and for E5).
"""
import argparse
import json
import math
from collections import defaultdict

import numpy as np
from common import RESULTS, load_results, merge, run_grid
from synthetic_setups import DENSE, REGIMES, TUNING_REGIMES

GRIDS = {
    "none": {"-": {}},
    "aux": {f"a{a:g}": dict(moe=dict(aux_loss_coef=a)) for a in (3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0)},
    "zloss": {f"z{z:g}": dict(moe=dict(z_loss_coef=z)) for z in (1e-4, 1e-3, 1e-2)},
    "explore_noise": {f"n{n:g}": dict(moe=dict(noise_std=n)) for n in (0.1, 0.3, 1.0, 3.0)},
    "auxfree_bias": {f"u{u:g}": dict(moe=dict(bias_update_rate=u)) for u in (1e-3, 3e-3, 1e-2, 3e-2, 1e-1)},
    "expert_choice": {"-": dict(moe=dict(router="expert_choice"))},  # capacity > 1 would add compute; kept at 1
    "sinkhorn": {f"i{i}": dict(moe=dict(router="sinkhorn", sinkhorn_iters=i)) for i in (1, 3, 10)},
    "cosine_router": {f"s{c}": dict(moe=dict(router_type="cosine", cosine_scale=float(c))) for c in (5, 10, 20, 40)},
    "revive_dead": {f"r{r}": dict(revive_every=r) for r in (50, 100, 200, 500)},
    "aux_then_release": {f"a{a:g}_t{t}": dict(moe=dict(aux_loss_coef=a), balance_until=t)
                         for a in (3e-2, 1e-1, 3e-1) for t in (50, 100, 300, 1000)},
    "bias_then_freeze": {f"u{u:g}_t{t}": dict(moe=dict(bias_update_rate=u), balance_until=t)
                         for u in (3e-3, 1e-2, 3e-2) for t in (100, 300, 1000)},
    "sinkhorn_then_release": {f"t{t}": dict(moe=dict(router="sinkhorn"), balance_until=t) for t in (100, 300, 1000)},
    "hash_fixed": {"-": dict(moe=dict(router="hash"))},
}
TUNE_SEEDS = [100, 101, 102, 103]
EVAL_SEEDS = list(range(10))
DENSE_WIDTHS = [8, 16, 32, 64, 128, 256, 512]
TUNED_PATH = RESULTS / "tuned_configs.json"


def _name(regime, method, tag):
    return f"{regime}__{method}__{tag}"


def tune_configs():
    return [merge(REGIMES[r], ov, dict(name=_name(r, m, tag), seed=s, decomp_probe=0))
            for r in TUNING_REGIMES for m, grid in GRIDS.items() for tag, ov in grid.items() for s in TUNE_SEEDS]


def select():
    res = load_results("e4_tune")
    val = defaultdict(list)
    for r in res:
        val[r["config"]["name"]].append(r["final"]["val_loss"])
    mean = {k: float(np.mean(v)) for k, v in val.items()}
    best_in_regime = {reg: min(v for k, v in mean.items() if k.startswith(f"{reg}__")) for reg in TUNING_REGIMES}
    out = {}
    for m, grid in GRIDS.items():
        per_regime = {}
        for reg in TUNING_REGIMES:
            tag = min(grid, key=lambda t: mean.get(_name(reg, m, t), math.inf))
            per_regime[reg] = {"tag": tag, "overrides": grid[tag], "val_loss": mean.get(_name(reg, m, tag))}

        def geo(tag):
            ratios = [mean.get(_name(reg, m, tag), math.inf) / best_in_regime[reg] for reg in TUNING_REGIMES]
            return float(np.exp(np.mean(np.log(ratios))))

        g = min(grid, key=geo)
        out[m] = {"global": {"tag": g, "overrides": grid[g], "geo_ratio": geo(g)}, "per_regime": per_regime,
                  "grid_val_loss": {t: {reg: mean.get(_name(reg, m, t)) for reg in TUNING_REGIMES} for t in grid}}
    TUNED_PATH.write_text(json.dumps(out, indent=1))
    print(f"wrote {TUNED_PATH}")
    for m, v in out.items():
        print(f"  {m:24s} global={v['global']['tag']:12s} geo={v['global']['geo_ratio']:.3f}  per-regime: "
              + ", ".join(f"{reg}={p['tag']}" for reg, p in v["per_regime"].items()))


def eval_configs():
    tuned = json.loads(TUNED_PATH.read_text())
    out = []
    for reg in TUNING_REGIMES:
        base = REGIMES[reg]
        for m, sel in tuned.items():
            g = sel["global"]
            out += [merge(base, g["overrides"], dict(name=f"{reg}__{m}__global", seed=s)) for s in EVAL_SEEDS]
            p = sel["per_regime"][reg]
            if p["tag"] != g["tag"]:
                out += [merge(base, p["overrides"], dict(name=f"{reg}__{m}__regime", seed=s)) for s in EVAL_SEEDS]
        out += [merge(base, dict(moe=dict(router="oracle")), dict(name=f"{reg}__oracle__-", seed=s)) for s in EVAL_SEEDS]
        out += [merge(base, dict(moe={**DENSE, "d_hidden": w}), dict(name=f"{reg}__dense_{w}__-", seed=s, decomp_probe=0))
                for w in DENSE_WIDTHS for s in EVAL_SEEDS]
    # Controls for validating redundancy metrics: same regime, known redundant (forced balance with 2 experts per
    # task) vs known non-redundant (free routing: one surviving expert per task).
    out += [merge(REGIMES["E16C8"], dict(moe=dict(aux_loss_coef=0.1)), dict(name="E16C8__control_redundant__-", seed=s))
            for s in EVAL_SEEDS]
    out += [merge(REGIMES["E16C8"], dict(name="E16C8__control_free__-", seed=s)) for s in EVAL_SEEDS]
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["tune", "select", "eval"])
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.stage == "tune":
        run_grid("synthetic", "e4_tune", tune_configs(), workers=a.workers)
    elif a.stage == "select":
        select()
    else:
        run_grid("synthetic", "e4_eval", eval_configs(), workers=a.workers)
