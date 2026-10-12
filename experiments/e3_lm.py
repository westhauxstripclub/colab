"""E3 - Do the synthetic conclusions hold in a real (small) language model?

Byte-level MoE transformer on Shakespeare / Python code / technical docs; ground truth for specialisation is the
domain of each sequence. Expert-choice routing is excluded: it lets experts pick tokens using the whole batch,
which leaks future tokens in an autoregressive model.

Presets:
  round1  2 layers, 8 experts x 128, 2500 steps (results/e3_lm, the first round)
  cpu     4 layers, 16 experts x 64 (top-2), 3000 steps, dense early logging        -> results/e3v2_lm_cpu
  gpu     6 layers, d256, 32 experts x 128 (top-2), seq 256, 10k steps              -> results/e3v2_lm_gpu
"""
import argparse

from common import ROOT, require_cuda, run_grid

PRESETS = {
    "round1": dict(exp="e3_lm", seeds=2, run=dict(steps=2500), moe=dict(n_experts=8, d_hidden=128, top_k=2)),
    "cpu": dict(exp="e3v2_lm_cpu", seeds=3,
                run=dict(n_layers=4, d_model=128, n_heads=4, seq_len=128, batch_size=32, steps=3000, eval_every=250,
                         early_eval_every=25, early_eval_until=500),
                moe=dict(n_experts=16, d_hidden=64, top_k=2)),
    "gpu": dict(exp="e3v2_lm_gpu", seeds=3,
                run=dict(n_layers=6, d_model=256, n_heads=8, seq_len=256, batch_size=32, steps=10_000, eval_every=500,
                         early_eval_every=50, early_eval_until=1000, lr=1e-3),
                moe=dict(n_experts=32, d_hidden=128, top_k=2)),
}


def methods(preset):
    P = PRESETS[preset]
    E, h, k = P["moe"]["n_experts"], P["moe"]["d_hidden"], P["moe"]["top_k"]
    dense = dict(n_experts=1, top_k=1, router="topk_renorm")
    release = P["run"].get("steps", 2500) // 10  # balance for the first 10% of training
    m = {
        "dense_active": dict(moe={**dense, "d_hidden": h * k}),  # same active params per token
        "dense_total": dict(moe={**dense, "d_hidden": h * E}),  # same total params
        "none": {},
        "aux_1e-2": dict(moe=dict(aux_loss_coef=1e-2)),
        "aux_1e-2_then_release": dict(moe=dict(aux_loss_coef=1e-2), balance_until=release),
        "auxfree_bias_1e-2": dict(moe=dict(bias_update_rate=1e-2)),
        "bias_then_freeze": dict(moe=dict(bias_update_rate=1e-2), balance_until=release),
        "hash_token": dict(moe=dict(router="hash")),
        "frozen_router": dict(freeze_router=True),
        "detach_input": dict(moe=dict(router_detach_input=True)),
        "frozen_and_detached": dict(freeze_router=True, moe=dict(router_detach_input=True)),
    }
    if preset == "round1":
        m = {
            "dense_active": m["dense_active"], "dense_total": m["dense_total"], "none": {},
            "aux_1e-2_batch": dict(moe=dict(aux_loss_coef=1e-2, aux_scope="batch")),
            "aux_1e-2_sequence": dict(moe=dict(aux_loss_coef=1e-2, aux_scope="sequence")),
            "aux_1e-2+zloss": dict(moe=dict(aux_loss_coef=1e-2, z_loss_coef=1e-3)),
            "auxfree_bias_1e-2": m["auxfree_bias_1e-2"], "hash_token": m["hash_token"],
            "frozen_router": m["frozen_router"],
        }
    return m


EXTRA_SEED_LIMITED = {"aux_3e-3": dict(moe=dict(aux_loss_coef=3e-3)), "aux_3e-2": dict(moe=dict(aux_loss_coef=3e-2))}


def configs(preset, seeds, data_dir, device="cpu", only=None):
    P = PRESETS[preset]
    out = []
    grid = dict(methods(preset))
    if preset != "round1":
        grid.update(EXTRA_SEED_LIMITED)
    for name, ov in grid.items():
        if only and name not in only:
            continue
        n_seeds = min(2, seeds) if name in EXTRA_SEED_LIMITED else seeds  # alpha sweep: fewer seeds
        moe = {**P["moe"], **ov.get("moe", {})}
        rest = {k: v for k, v in ov.items() if k != "moe"}
        for s in range(n_seeds):
            out.append(dict(name=name, seed=s, data_dir=data_dir, device=device, moe=moe, **P["run"], **rest))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=list(PRESETS), default="cpu")
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--methods", nargs="*", default=None, help="subset of method names to run")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--data_dir", default=str(ROOT / "data"))
    ap.add_argument("--device", default="cpu",
                    help="'cuda' on a GPU; with --workers N each worker gets its own GPU (cuda:0, cuda:1, ...)")
    a = ap.parse_args()
    require_cuda(a.device)
    P = PRESETS[a.preset]
    run_grid("lm", P["exp"], configs(a.preset, a.seeds or P["seeds"], a.data_dir, a.device, a.methods),
             workers=a.workers, threads=a.threads)
