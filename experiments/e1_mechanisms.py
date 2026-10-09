"""E1 - WHY do experts collapse? Causal interventions on the router/expert feedback loop.

Each condition removes or amplifies one link of the hypothesised loop
    expert gets tokens -> expert gets gradient -> expert improves on those tokens
    -> router (which only sees feedback from the expert it picked) sends it more tokens
and we measure how many experts end up dead. Plus a data-scarce regime that directly tests the
"popular experts are overfit" hypothesis.
"""
import argparse

from common import merge, run_grid
from synthetic_setups import REGIMES

CONDITIONS = {
    "baseline": {},
    "frozen_router": dict(freeze_router=True),  # break link: router -> more tokens
    "frozen_experts": dict(freeze_experts=True),  # break link: tokens -> expert improves
    "identical_init": dict(moe=dict(identical_init=True)),  # remove initial expert asymmetry
    "router_lr_x0.1": dict(router_lr_mult=0.1),  # slower router relative to experts
    "router_lr_x10": dict(router_lr_mult=10.0),  # faster router relative to experts
    "adam_load_normalized": dict(load_normalized_grads=True),
    "sgd": dict(optimizer="sgd", lr=0.05),  # update size proportional to #tokens
    "sgd_load_normalized": dict(optimizer="sgd", lr=0.05, load_normalized_grads=True),
    "head_start_200": dict(head_start_steps=200),  # give expert 0 a lead
    "router_warmup_100": dict(router_warmup_steps=100),  # let experts learn before the router reacts
    "zero_init_output": dict(moe=dict(zero_init_output=True)),  # no random expert outputs at init
    "topk_renorm_k1": dict(moe=dict(router="topk_renorm")),  # Mixtral-style gate with k=1: no router gradient
}

OVERFIT = dict(moe=dict(d_hidden=32, n_experts=8), task=dict(n_clusters=8, n_train=800, n_val=4000))


def configs(seeds):
    out = []
    for regime in ("E8", "E32"):
        base = REGIMES[regime]
        for cond, ov in CONDITIONS.items():
            for s in seeds:
                out.append(merge(base, ov, dict(name=f"{regime}__{cond}", seed=s)))
    # More experts than latent tasks: the realistic case where "one expert per skill" is impossible.
    for s in seeds:
        out.append(merge(REGIMES["E16C8"], dict(name="E16C8__baseline", seed=s)))
    # Data-scarce regime where overfitting is actually possible (100 examples per latent task).
    for cond, ov in {"baseline": {}, "aux_1e-1": dict(moe=dict(aux_loss_coef=0.1))}.items():
        for s in seeds:
            out.append(merge(OVERFIT, ov, dict(name=f"overfit__{cond}", seed=s)))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    run_grid("synthetic", "e1_mechanisms", configs(range(a.seeds)), workers=a.workers)
