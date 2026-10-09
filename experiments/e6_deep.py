"""E6 - Depth and the representation pathway.

Deep residual MoE stacks on the synthetic tasks, where every router sees learned representations. Two task
families:
  clusters  tasks linearly separable in the input (E16, C16), depth 1, 2, 4
  xor       tasks invisible to any linear function of the input (E8, C8), depth 2 and 4 with a dense pre-block that
            can learn the needed features, plus depth 2 without it (routing cannot see the tasks at all)
Interventions separate the pathways: freezing the router removes the router-weight pathway; detaching the router's
input removes the routing gradient's pull on the representations (incidental drift from the task loss remains);
both together leave only incidental drift. Logit decomposition is logged throughout training.
"""
import argparse
import json

from common import merge, run_grid
from e4_tuning import TUNED_PATH

BIG = dict(n_train=16000, n_val=8000, n_test=8000)
FAMILIES = {
    "clu_d1": dict(depth=1, moe=dict(d_hidden=8, n_experts=16), task=dict(n_clusters=16, **BIG)),
    "clu_d2": dict(depth=2, moe=dict(d_hidden=8, n_experts=16), task=dict(n_clusters=16, **BIG)),
    "clu_d4": dict(depth=4, moe=dict(d_hidden=8, n_experts=16), task=dict(n_clusters=16, **BIG)),
    "xor_d2": dict(depth=2, pre_dense=64, moe=dict(d_hidden=8, n_experts=8), task=dict(n_clusters=8, routing="xor")),
    "xor_d4": dict(depth=4, pre_dense=64, moe=dict(d_hidden=8, n_experts=8), task=dict(n_clusters=8, routing="xor")),
    "xor_d2_nopre": dict(depth=2, moe=dict(d_hidden=8, n_experts=8), task=dict(n_clusters=8, routing="xor")),
}


def conditions():
    tuned = json.loads(TUNED_PATH.read_text())
    g = {m: tuned[m]["global"]["overrides"] for m in ("aux", "auxfree_bias", "aux_then_release", "cosine_router")}
    return {
        "none": {},
        "frozen_router": dict(freeze_router=True),
        "detach_input": dict(moe=dict(router_detach_input=True)),
        "frozen_and_detached": dict(freeze_router=True, moe=dict(router_detach_input=True)),
        "cosine_router": g["cosine_router"],
        "aux": g["aux"],
        "auxfree_bias": g["auxfree_bias"],
        "aux_then_release": g["aux_then_release"],
        "oracle": dict(moe=dict(router="oracle")),
    }


def configs(seeds):
    conds = conditions()
    out = []
    for fam, base in FAMILIES.items():
        names = ["none", "oracle"] if fam == "xor_d2_nopre" else list(conds)
        for c in names:
            out += [merge(base, conds[c], dict(name=f"{fam}__{c}", seed=s)) for s in seeds]
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    run_grid("synthetic", "e6_deep", configs(list(range(a.seeds))), workers=a.workers)
