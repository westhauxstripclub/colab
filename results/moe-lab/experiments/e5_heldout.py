"""E5 - Pre-registered held-out test (see PREREGISTRATION.md, committed before this was run).

Seven regimes never used for designing or tuning anything. Every method runs with the single *global* config chosen
in E4 on the round-1 regimes (no per-regime tuning is possible on held-out data). Scored on the test split of 10
fresh evaluation seeds.
"""
import argparse
import json

from common import merge, run_grid
from e4_tuning import EVAL_SEEDS, TUNED_PATH
from synthetic_setups import DENSE

BIG = dict(n_train=16000, n_val=8000, n_test=8000)
HELDOUT = {
    "E16C16k2": dict(moe=dict(d_hidden=8, n_experts=16, top_k=2), task=dict(n_clusters=16, **BIG)),  # top-2 routing
    "E32C16": dict(moe=dict(d_hidden=8, n_experts=32), task=dict(n_clusters=16, **BIG)),  # 2 experts per skill
    "E8C24": dict(moe=dict(d_hidden=8, n_experts=8), task=dict(n_clusters=24, n_train=24000, n_val=8000, n_test=8000)),
    "E16C16zipf08": dict(moe=dict(d_hidden=8, n_experts=16), task=dict(n_clusters=16, freq="zipf", zipf_s=0.8, **BIG)),
    "XOR8deep2": dict(depth=2, pre_dense=64, moe=dict(d_hidden=8, n_experts=8), task=dict(n_clusters=8, routing="xor")),
    "CE16": dict(moe=dict(d_hidden=8, n_experts=16), task=dict(n_clusters=16, loss="ce", **BIG)),
    "deep3E16C16": dict(depth=3, moe=dict(d_hidden=8, n_experts=16), task=dict(n_clusters=16, **BIG)),
}


def configs(seeds):
    tuned = json.loads(TUNED_PATH.read_text())
    out = []
    for reg, base in HELDOUT.items():
        E, k, h = base["moe"]["n_experts"], base["moe"].get("top_k", 1), base["moe"]["d_hidden"]
        for m, sel in tuned.items():
            out += [merge(base, sel["global"]["overrides"], dict(name=f"{reg}__{m}", seed=s)) for s in seeds]
        out += [merge(base, dict(moe=dict(router="oracle", top_k=1)), dict(name=f"{reg}__oracle", seed=s)) for s in seeds]
        for label, width in (("dense_active", h * k), ("dense_total", h * E)):
            out += [merge(base, dict(moe={**DENSE, "d_hidden": width}), dict(name=f"{reg}__{label}", seed=s, decomp_probe=0))
                    for s in seeds]
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    run_grid("synthetic", "e5_heldout", configs(EVAL_SEEDS), workers=a.workers)
