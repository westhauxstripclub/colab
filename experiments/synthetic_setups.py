"""Shared synthetic regimes: <experts>/<latent tasks>. E8 = 8/8, E32 = 32/32, E16C8 = 16 experts for 8 tasks."""
REGIMES = {
    "E8": dict(moe=dict(d_hidden=8, n_experts=8), task=dict(n_clusters=8, n_train=8000, n_val=4000)),
    "E32": dict(moe=dict(d_hidden=8, n_experts=32), task=dict(n_clusters=32, n_train=32000, n_val=8000)),
    "E16C8": dict(moe=dict(d_hidden=8, n_experts=16), task=dict(n_clusters=8, n_train=8000, n_val=4000)),
}
DENSE = dict(n_experts=1, top_k=1, router="topk_renorm")  # one expert with gate 1 == a plain MLP
