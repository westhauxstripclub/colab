"""Correctness checks for the routing math the experiments rely on."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from moe_lab.metrics import load_stats, specialization  # noqa: E402
from moe_lab.moe import MoEConfig, MoELayer, sinkhorn_scores  # noqa: E402


def make(**kw):
    torch.manual_seed(0)
    return MoELayer(MoEConfig(d_in=6, d_out=3, d_hidden=5, **{"n_experts": 4, **kw}))


def test_aux_loss_is_one_at_perfect_balance():
    layer = make(aux_loss_coef=1.0)
    with torch.no_grad():
        layer.router.weight.zero_()  # uniform router probabilities
    M = torch.eye(4, dtype=torch.bool).repeat(8, 1)  # every expert gets exactly 1/4 of tokens
    P = torch.full((32, 4), 0.25)
    assert layer._switch_aux(M, P, None).item() == pytest.approx(1.0)


def test_unselected_experts_get_zero_gradient():
    layer = make(top_k=1)
    x = torch.randn(64, 6)
    layer(x).pow(2).sum().backward()
    used = layer.last["M"].any(0)
    for p in layer.experts.params():
        g = p.grad.flatten(1).abs().sum(1)
        assert torch.all(g[~used] == 0)
        assert torch.all(g[used] > 0)


def test_topk_renorm_k1_router_gets_no_task_gradient():
    layer = make(top_k=1, router="topk_renorm")
    layer(torch.randn(64, 6)).pow(2).sum().backward()
    assert layer.router.weight.grad is None or torch.all(layer.router.weight.grad == 0)


def test_softmax_topk_router_gets_gradient_on_every_logit():
    layer = make(top_k=1, router="softmax_topk")
    layer(torch.randn(64, 6)).pow(2).sum().backward()
    assert torch.all(layer.router.weight.grad.abs().sum(1) > 0)


def test_expert_choice_is_balanced_by_construction():
    layer = make(router="expert_choice", top_k=1)
    layer(torch.randn(64, 6))
    assert layer.last["M"].sum(0).tolist() == [16, 16, 16, 16]


def test_capacity_factor_caps_tokens_per_expert():
    layer = make(top_k=1, capacity_factor=1.0)
    with torch.no_grad():
        layer.router.weight.zero_()
        layer.router.weight[0] = 5.0  # everyone wants expert 0
    layer(torch.rand(64, 6) + 0.1)
    assert layer.last["M"].sum(0)[0].item() == 16
    assert layer.last["M_sel"].sum(0)[0].item() == 64


def test_bias_update_moves_toward_balance():
    layer = make(top_k=1, bias_update_rate=0.1)
    with torch.no_grad():
        layer.router.weight.zero_()
        layer.router.weight[0] = 5.0
    layer(torch.rand(64, 6) + 0.1)
    layer.update_bias()
    assert layer.bias[0] < 0 and torch.all(layer.bias[1:] > 0)


def test_sinkhorn_selection_is_near_balanced():
    torch.manual_seed(0)
    logits = torch.randn(400, 4) + torch.tensor([3.0, 0, 0, 0])
    sel = sinkhorn_scores(logits, 20).argmax(-1)
    counts = torch.bincount(sel, minlength=4).float()
    assert counts.max() / counts.min() < 1.5


def test_sparse_and_dense_compute_paths_agree():
    x = torch.randn(32, 6)
    outs, grads = [], []
    for compute_all in (True, False):
        layer = make(top_k=2, compute_all=compute_all)
        y = layer(x)
        y.pow(2).sum().backward()
        outs.append(y.detach())
        grads.append(layer.experts.w1.grad.clone())
    assert torch.allclose(outs[0], outs[1], atol=1e-5)
    assert torch.allclose(grads[0], grads[1], atol=1e-5)


def test_identical_init_copies_expert_zero():
    layer = make(identical_init=True)
    assert torch.equal(layer.experts.w1[0], layer.experts.w1[3])


def test_metrics_basics():
    M = torch.zeros(10, 4, dtype=torch.bool)
    M[:, 0] = True
    st = load_stats(M)
    assert st["effective_experts"] == pytest.approx(1.0) and st["dead_experts"] == 3
    labels = np.repeat(np.arange(4), 25)
    assert specialization(labels, labels)["label_info_recovered"] == pytest.approx(1.0)
    assert specialization(np.zeros(100, int), labels)["label_info_recovered"] == pytest.approx(0.0)


def test_aux_loss_pull_on_a_dead_expert_vanishes_with_its_probability():
    """d L_aux / d logit_i = E * P_i * (f_i - sum_j f_j P_j): for a dead expert (f_i = 0) the restoring push is
    proportional to P_i, so once the router is confident (P_i ~ 0) the aux loss can no longer revive it."""
    layer = make(top_k=1, aux_loss_coef=1.0)
    pulls = []
    for dead_logit in (0.0, -3.0, -6.0):
        logits = torch.tensor([[2.0, 1.0, 0.5, dead_logit]] * 16, requires_grad=True)
        M = torch.zeros(16, 4, dtype=torch.bool)
        M[:8, 0] = M[8:12, 1] = M[12:, 2] = True  # expert 3 receives no tokens
        layer._switch_aux(M, logits.softmax(-1), None).backward()
        pulls.append(-logits.grad[:, 3].sum().item())
    assert all(p > 0 for p in pulls)  # gradient descent raises the dead expert's logit...
    assert pulls[0] > 10 * pulls[2]  # ...but the push shrinks with its probability


# ----------------------------------------------------------------------------------- round 2 additions
from moe_lab.data import make_cluster_task  # noqa: E402
from moe_lab.metrics import logit_decomposition  # noqa: E402
from moe_lab.stats import bootstrap_ci, compare, holm, sign_flip_pvalue  # noqa: E402
from moe_lab.train_synthetic import SynthRunConfig, _rownorm_router_grads, run_synthetic  # noqa: E402


def _router_only_layer(detach):
    layer = make(top_k=1, router_detach_input=detach)
    with torch.no_grad():
        layer.experts.w1.zero_()  # expert outputs no longer depend on x: x reaches the output only via the router
        layer.experts.b1.normal_()  # ...but are nonzero constants
    return layer


def test_router_detach_input_blocks_routing_gradient_into_representations():
    for detach, expect_x_grad in ((False, True), (True, False)):
        layer = _router_only_layer(detach)
        x = torch.randn(32, 6, requires_grad=True)
        layer(x).pow(2).sum().backward()
        x_grad = x.grad is not None and x.grad.abs().sum() > 0
        assert x_grad == expect_x_grad
        assert layer.router.weight.grad.abs().sum() > 0  # the router itself still learns


def test_xor_tasks_are_not_linearly_decodable():
    t = make_cluster_task(n_clusters=8, routing="xor", seed=0)
    bit0 = (t.c_train % 2).float()
    w = torch.zeros(t.x_train.shape[1] + 1, requires_grad=True)
    opt = torch.optim.Adam([w], lr=0.05)
    X = torch.cat([t.x_train, torch.ones(len(t.x_train), 1)], 1)
    for _ in range(300):
        opt.zero_grad()
        F.binary_cross_entropy_with_logits(X @ w, bit0).backward()
        opt.step()
    Xv = torch.cat([t.x_val, torch.ones(len(t.x_val), 1)], 1)
    acc = ((Xv @ w > 0).float() == (t.c_val % 2).float()).float().mean().item()
    assert acc < 0.56  # chance is 0.5; a linear probe cannot recover an XOR bit


def test_test_split_is_separate_and_ce_labels_are_classes():
    t = make_cluster_task(loss="ce", d_out=5, seed=1)
    assert t.x_test.shape == t.x_val.shape and not torch.equal(t.x_test, t.x_val)
    assert t.y_train.dtype == torch.long and int(t.y_train.max()) < 5


def test_deep_and_ce_runs_report_per_layer_and_test_metrics():
    r = run_synthetic(SynthRunConfig(name="t", depth=2, steps=20, eval_every=10, decomp_probe=128,
                                     moe=dict(d_hidden=4), task=dict(loss="ce")))
    f = r["final"]
    assert len(f["layers"]) == 2 and 0.0 <= f["test_acc"] <= 1.0
    assert f["dead_experts"] == pytest.approx(np.mean([l["dead_experts"] for l in f["layers"]]))
    assert len(r["history"][-1]["decomp"]) == 2


def test_logit_decomposition_attributes_change_to_the_right_pathway():
    layer = make(top_k=1)
    x0 = torch.randn(64, 6)
    p0 = {k: v.detach().clone() for k, v in layer.router.named_parameters()}
    moved_inputs = logit_decomposition(layer, x0, x0 + torch.randn(64, 6), p0, 1)
    assert moved_inputs["rms_router"] == 0 and moved_inputs["rms_rep"] > 0
    with torch.no_grad():
        layer.router.weight.add_(torch.randn_like(layer.router.weight))
    moved_router = logit_decomposition(layer, x0, x0, p0, 1)
    assert moved_router["rms_rep"] == 0 and moved_router["rms_router"] == pytest.approx(moved_router["rms_total"])


def test_router_grad_rownorm_equalises_row_norms():
    layer = make(top_k=1)
    layer(torch.randn(64, 6)).pow(2).sum().backward()
    _rownorm_router_grads(layer)
    norms = layer.router.weight.grad.norm(dim=1)
    assert torch.allclose(norms[norms > 0], norms[norms > 0].mean().expand_as(norms[norms > 0]), rtol=1e-5)


def test_paired_statistics():
    rng = np.random.default_rng(0)
    d = 1.0 + 0.1 * rng.standard_normal(20)
    mean, lo, hi = bootstrap_ci(d)
    assert lo < 1.0 < hi and lo < mean < hi
    assert sign_flip_pvalue(d) < 0.001 and sign_flip_pvalue(rng.standard_normal(20) * 0.0) == 1.0
    adj = holm([0.01, 0.04, 0.03])
    assert all(a >= p for a, p in zip(adj, [0.01, 0.04, 0.03])) and adj[0] == pytest.approx(0.03)
    res = compare({s: 2.0 + s for s in range(10)}, {s: 1.0 + s for s in range(12)})
    assert res["n"] == 10 and res["diff"] == pytest.approx(1.0)


def test_real_checkpoint_selection_is_distinct_sorted_and_spans_training():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    from e7_real_checkpoints import pick_checkpoints, saturation

    names = ["main"] + [f"step{s}-tokens{s // 250}B" for s in range(0, 1_300_000, 5000)]
    chosen = pick_checkpoints(names, 6)
    steps = [int(c.split("-")[0][4:]) for c in chosen]
    assert len(set(chosen)) == 6 and steps == sorted(steps) and steps[0] == 0 and steps[-1] == 1_295_000
    top = torch.tensor([[0, 1], [2, 3]])
    assert saturation(top, top, 2) == 1.0 and saturation(top, torch.tensor([[1, 5], [6, 7]]), 2) == 0.25


def test_single_expert_dense_baseline_runs_with_all_diagnostics():
    r = run_synthetic(SynthRunConfig(name="t", steps=5, eval_every=5, snapshot_steps=(0,),
                                     moe=dict(n_experts=1, top_k=1, router="topk_renorm", d_hidden=4)))
    assert math.isnan(r["final"]["substitution_gap"])
