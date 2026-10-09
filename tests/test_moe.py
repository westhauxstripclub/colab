"""Correctness checks for the routing math the experiments rely on."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

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
