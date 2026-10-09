"""E3 - Do the synthetic conclusions hold in a real (tiny) language model?

Byte-level 2-layer MoE transformer on Shakespeare / Python code / technical docs. Ground truth for
specialisation = the domain of each sequence. Expert-choice routing is excluded: it lets experts pick
tokens using the whole batch, which leaks future tokens in an autoregressive model.
"""
import argparse

from common import ROOT, run_grid

DENSE = dict(n_experts=1, top_k=1, router="topk_renorm")
METHODS = {
    "dense_active": dict(moe={**DENSE, "d_hidden": 256}),  # same active params as top-2 of 8x128
    "dense_total": dict(moe={**DENSE, "d_hidden": 1024}),  # same total params as 8x128
    "none": {},
    "aux_1e-2_batch": dict(moe=dict(aux_loss_coef=1e-2, aux_scope="batch")),
    "aux_1e-2_sequence": dict(moe=dict(aux_loss_coef=1e-2, aux_scope="sequence")),
    "aux_1e-2+zloss": dict(moe=dict(aux_loss_coef=1e-2, z_loss_coef=1e-3)),
    "auxfree_bias_1e-2": dict(moe=dict(bias_update_rate=1e-2)),
    "hash_token": dict(moe=dict(router="hash")),
    "frozen_router": dict(freeze_router=True),
}


def configs(seeds, steps, data_dir, device="cpu"):
    return [dict(name=m, seed=s, steps=steps, data_dir=data_dir, device=device, **ov)
            for m, ov in METHODS.items() for s in seeds]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--data_dir", default=str(ROOT / "data"))
    ap.add_argument("--device", default="cpu", help="'cuda' on a Colab GPU (use --workers 1)")
    a = ap.parse_args()
    run_grid("lm", "e3_lm", configs(range(a.seeds), a.steps, a.data_dir, a.device), workers=a.workers, threads=a.threads)
