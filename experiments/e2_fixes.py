"""E2 - What FIXES collapse, and does balanced load actually buy specialisation and quality?

Compares balancing methods against two references: dense MLPs of increasing width (to express each
MoE as a "dense-equivalent" size) and an oracle router that routes by the ground-truth task id.
"""
import argparse

from common import merge, run_grid
from synthetic_setups import DENSE, REGIMES

METHODS = {
    "none": {},
    "aux_1e-3": dict(moe=dict(aux_loss_coef=1e-3)),
    "aux_1e-2": dict(moe=dict(aux_loss_coef=1e-2)),
    "aux_1e-1": dict(moe=dict(aux_loss_coef=1e-1)),
    "aux_1": dict(moe=dict(aux_loss_coef=1.0)),
    "zloss_1e-3": dict(moe=dict(z_loss_coef=1e-3)),
    "aux_1e-2+zloss": dict(moe=dict(aux_loss_coef=1e-2, z_loss_coef=1e-3)),
    "explore_noise_1": dict(moe=dict(noise_std=1.0)),
    "auxfree_bias_1e-2": dict(moe=dict(bias_update_rate=1e-2)),
    "expert_choice": dict(moe=dict(router="expert_choice")),
    "sinkhorn": dict(moe=dict(router="sinkhorn")),
    "cosine_router": dict(moe=dict(router_type="cosine")),
    "revive_dead": dict(revive_every=200),
    "hash_fixed": dict(moe=dict(router="hash")),
    "oracle": dict(moe=dict(router="oracle")),
}
ZIPF_METHODS = ["none", "aux_1e-2", "aux_1e-1", "auxfree_bias_1e-2", "expert_choice", "sinkhorn", "revive_dead", "oracle"]
DENSE_WIDTHS = [8, 16, 32, 64, 128, 256, 512]


def configs(seeds):
    out = []
    for regime in ("E8", "E32"):
        base = REGIMES[regime]
        for m, ov in METHODS.items():
            for s in seeds:
                out.append(merge(base, ov, dict(name=f"{regime}__{m}", seed=s)))
        for w in DENSE_WIDTHS:
            for s in seeds[:4]:
                out.append(merge(base, dict(moe={**DENSE, "d_hidden": w}), dict(name=f"{regime}__dense_{w}", seed=s)))
    for m in [m for m in ZIPF_METHODS if m != "oracle"] + ["explore_noise_1", "hash_fixed"]:
        for s in seeds:
            out.append(merge(REGIMES["E16C8"], METHODS[m], dict(name=f"E16C8__{m}", seed=s)))
    for w in DENSE_WIDTHS:
        for s in seeds[:4]:
            out.append(merge(REGIMES["E16C8"], dict(moe={**DENSE, "d_hidden": w}), dict(name=f"E16C8__dense_{w}", seed=s)))
    zipf = merge(REGIMES["E8"], dict(task=dict(freq="zipf", zipf_s=1.2)))
    for m in ZIPF_METHODS:
        for s in seeds:
            out.append(merge(zipf, METHODS[m], dict(name=f"E8zipf__{m}", seed=s)))
    for w in DENSE_WIDTHS:
        for s in seeds[:4]:
            out.append(merge(zipf, dict(moe={**DENSE, "d_hidden": w}), dict(name=f"E8zipf__dense_{w}", seed=s)))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    run_grid("synthetic", "e2_fixes", configs(list(range(a.seeds))), workers=a.workers)
