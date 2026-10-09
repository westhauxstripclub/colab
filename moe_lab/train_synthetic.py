"""Training + diagnostics on the synthetic mixture-of-tasks problem (ground-truth clusters known)."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from .data import make_cluster_task
from .metrics import entropy_bits, expert_redundancy, gini, load_stats, router_stats, specialization, spearman
from .models import SyntheticMoE
from .moe import TOKEN_CHOICE_ROUTERS, MoEConfig

SYNTH_MOE_DEFAULTS = dict(d_hidden=16, n_experts=8, top_k=1, router="softmax_topk")


@dataclass
class SynthRunConfig:
    name: str
    seed: int = 0
    moe: dict = field(default_factory=dict)  # overrides for MoEConfig
    task: dict = field(default_factory=dict)  # overrides for make_cluster_task
    steps: int = 3000
    batch_size: int = 256
    lr: float = 3e-3
    optimizer: str = "adam"  # "adam" | "sgd"
    sgd_momentum: float = 0.9
    router_lr_mult: float = 1.0  # router learning rate relative to the experts
    # --- causal interventions on the feedback loop
    freeze_router: bool = False  # routing fixed at its random init
    router_warmup_steps: int = 0  # router frozen for the first N steps while the experts learn
    freeze_experts: bool = False  # experts fixed at init; only the router learns (pure selection)
    load_normalized_grads: bool = False  # rescale each expert's grad by mean_load / load_e
    head_start_steps: int = 0  # train expert 0 alone on everything before MoE training begins
    # --- fix: balance only while the early race is decided, then let routing settle freely
    balance_until: int = 0  # >0: at this step drop the aux loss, freeze the bias, switch Sinkhorn -> softmax top-k
    # --- fix: re-initialise starved experts by splitting the busiest one
    revive_every: int = 0
    revive_threshold: float = 0.1  # "dead" = EMA load below threshold / E
    eval_every: int = 100  # plus dense logging (every 10 steps) over the first 300, where collapse happens
    snapshot_steps: tuple = (0, 50, 200, 1000)


def _mse_per_token(pred, y):
    return (pred - y).pow(2).mean(-1)


@torch.no_grad()
def light_eval(model, task, step):
    model.eval()
    layer = model.moe
    pred = model(task.x_val, labels=task.c_val)
    val = F.mse_loss(pred, task.y_val).item()
    st = {"step": step, "val_loss": val}
    st.update(load_stats(layer.last["M"]))
    st.update(router_stats(layer.last["logits"]))
    top1 = layer.last["G"].argmax(-1).numpy()
    st.update(specialization(top1, task.c_val.numpy()))
    n = min(len(task.x_val), len(task.x_train))
    st["train_loss"] = F.mse_loss(model(task.x_train[:n], labels=task.c_train[:n]), task.y_train[:n]).item()
    if layer.cfg.bias_update_rate > 0:
        st["bias"] = layer.bias.tolist()
    model.train()
    return st


@torch.no_grad()
def heavy_eval(model, task):
    """Counterfactual diagnostics: what would each expert do on each cluster?"""
    model.eval()
    layer = model.moe
    E, C = layer.cfg.n_experts, task.n_clusters
    x, y, c = task.x_val, task.y_val, task.c_val
    pred = model(x, labels=c)
    tok_loss = _mse_per_token(pred, y)
    M, G = layer.last["M"], layer.last["G"]
    top1 = G.argmax(-1)

    cf = layer.counterfactual_outputs(x)  # [N, E, d]
    cf_loss = (cf - y.unsqueeze(1)).pow(2).mean(-1)  # [N, E]
    expert_cluster_loss = torch.stack([cf_loss[c == k].mean(0) for k in range(C)], 1)  # [E, C]
    cluster_to_expert = torch.stack([M[c == k].float().mean(0) for k in range(C)])  # [C, E]
    oracle = cf_loss.min(-1).values

    load = M.float().sum(0).numpy()
    alive = load / max(load.sum(), 1) >= 0.1 / E
    out = {
        "expert_cluster_loss": expert_cluster_loss.tolist(),
        "cluster_to_expert": cluster_to_expert.tolist(),
        "per_cluster_val_loss": [tok_loss[c == k].mean().item() for k in range(C)],
        "best_expert_loss": oracle.mean().item(),
        "routing_regret": (tok_loss - oracle).mean().item() if layer.cfg.top_k == 1 else float("nan"),
        "frac_tokens_on_best_expert": (cf_loss.argmin(-1) == top1).float().mean().item(),
    }
    raw = layer.experts.forward_all(x[:2000])
    out.update({f"all_{k}": v for k, v in expert_redundancy(raw).items()})
    out.update({f"alive_{k}": v for k, v in expert_redundancy(raw, alive).items()})
    model.train()
    return out


@torch.no_grad()
def overfitting_eval(model, task):
    """Is a heavily used expert *overfit*? Compare train vs val loss on the tokens routed to each expert."""
    model.eval()
    layer = model.moe
    E = layer.cfg.n_experts
    res = {}
    for split, x, y, c in (("train", task.x_train, task.y_train, task.c_train), ("val", task.x_val, task.y_val, task.c_val)):
        pred = model(x, labels=c)
        tl = _mse_per_token(pred, y)
        top1 = layer.last["G"].argmax(-1)
        has = layer.last["M"].any(-1)
        res[split] = [(tl[(top1 == e) & has].mean().item() if ((top1 == e) & has).sum() >= 20 else float("nan")) for e in range(E)]
        res[f"{split}_count"] = [int(((top1 == e) & has).sum()) for e in range(E)]
    gap = [v - t for t, v in zip(res["train"], res["val"])]
    load = np.array(res["train_count"], dtype=float)
    ok = ~np.isnan(gap)
    model.train()
    return {
        "per_expert_train_loss": res["train"],
        "per_expert_val_loss": res["val"],
        "per_expert_gap": gap,
        "per_expert_train_tokens": res["train_count"],
        "spearman_load_vs_gap": spearman(load[ok], np.array(gap)[ok]),
    }


@torch.no_grad()
def ablation_eval(model, task):
    """Loss increase when each expert is removed from routing (tokens fall back to the next choice)."""
    model.eval()
    layer = model.moe
    base = F.mse_loss(model(task.x_val, labels=task.c_val), task.y_val).item()
    deltas = []
    for e in range(layer.cfg.n_experts):
        layer.disabled = [e]
        deltas.append(F.mse_loss(model(task.x_val, labels=task.c_val), task.y_val).item() - base)
    layer.disabled = []
    share = np.clip(deltas, 0, None)
    model.train()
    return {
        "ablation_delta": deltas,
        "ablation_gini": gini(share),
        "useful_experts": float(2 ** entropy_bits(share)) if share.sum() > 0 else 0.0,
    }


def _scale_expert_grads(layer):
    load = layer.last["M"].float().sum(0)
    scale = load.mean() / load.clamp_min(1.0)
    for p in layer.experts.params():
        if p.grad is not None:
            p.grad.mul_(scale.view(-1, *[1] * (p.dim() - 1)))


@torch.no_grad()
def _revive_dead(layer, opt, ema_load, threshold):
    E = layer.cfg.n_experts
    revived = []
    for d in (ema_load < threshold / E).nonzero(as_tuple=True)[0].tolist():
        b = int(ema_load.argmax())
        tensors = list(layer.experts.params())
        if layer.cfg.router_type == "linear":
            tensors.append(layer.router.weight)
        for p in tensors:
            p[d] = p[b] + 0.01 * p[b].std() * torch.randn_like(p[b])
            st = opt.state.get(p, {})
            if "exp_avg" in st:
                st["exp_avg"][d] = 0
                st["exp_avg_sq"][d] = st["exp_avg_sq"][b]
            if "momentum_buffer" in st and st["momentum_buffer"] is not None:
                st["momentum_buffer"][d] = 0
        layer.bias[d] = layer.bias[b]
        ema_load[b] = ema_load[d] = ema_load[b] / 2
        revived.append(d)
    return revived


def run_synthetic(rc: SynthRunConfig) -> dict:
    t0 = time.time()
    torch.manual_seed(rc.seed)
    task = make_cluster_task(seed=rc.seed, **rc.task)
    d_in, d_out = task.x_train.shape[1], task.y_train.shape[1]
    cfg = MoEConfig(d_in=d_in, d_out=d_out, **{**SYNTH_MOE_DEFAULTS, **rc.moe})
    model = SyntheticMoE(cfg)
    layer = model.moe
    if rc.freeze_router:
        layer.router.requires_grad_(False)
    if rc.freeze_experts:
        layer.experts.requires_grad_(False)
    router_params = [p for p in layer.router.parameters() if p.requires_grad]
    expert_params = [p for p in layer.experts.parameters() if p.requires_grad]
    groups = [g for g in ({"params": expert_params}, {"params": router_params, "lr": rc.lr * rc.router_lr_mult}) if g["params"]]
    if rc.optimizer == "adam":
        opt = torch.optim.Adam(groups, lr=rc.lr)
    else:
        opt = torch.optim.SGD(groups, lr=rc.lr, momentum=rc.sgd_momentum)
    gen = torch.Generator().manual_seed(rc.seed)
    n_train = len(task.x_train)

    def train_step():
        idx = torch.randint(0, n_train, (rc.batch_size,), generator=gen)
        x, y, c = task.x_train[idx], task.y_train[idx], task.c_train[idx]
        loss = F.mse_loss(model(x, labels=c), y) + layer.aux_loss()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if layer.force_expert is None and train_step.n < rc.router_warmup_steps:
            for p in router_params:
                p.grad = None
        train_step.n += 1
        if rc.load_normalized_grads:
            _scale_expert_grads(layer)
        opt.step()
        layer.update_bias()

    train_step.n = 0
    if rc.head_start_steps:
        layer.force_expert = 0
        for _ in range(rc.head_start_steps):
            train_step()
        layer.force_expert = None

    history, snapshots, revivals = [], {}, []
    ema_load = torch.full((cfg.n_experts,), 1.0 / cfg.n_experts)
    for step in range(rc.steps + 1):
        if step % rc.eval_every == 0 or step == rc.steps or (step < 300 and step % 10 == 0):
            history.append(light_eval(model, task, step))
        if step in rc.snapshot_steps:
            snapshots[step] = heavy_eval(model, task)
        if step == rc.steps:
            break
        if rc.balance_until and step == rc.balance_until:
            layer.cfg.aux_loss_coef = 0.0
            layer.bias_frozen = True
            if layer.cfg.router == "sinkhorn":
                layer.cfg.router = "softmax_topk"
        train_step()
        if rc.revive_every and cfg.router in TOKEN_CHOICE_ROUTERS:
            frac = layer.last["M"].float().sum(0)
            ema_load = 0.95 * ema_load + 0.05 * frac / frac.sum()
            if (step + 1) % rc.revive_every == 0 and step + 1 < 0.8 * rc.steps:
                for d in _revive_dead(layer, opt, ema_load, rc.revive_threshold):
                    revivals.append((step + 1, d))

    final = {**history[-1], **heavy_eval(model, task), **overfitting_eval(model, task), **ablation_eval(model, task)}
    return {
        "config": asdict(rc),
        "moe_config": asdict(cfg),
        "history": history,
        "snapshots": {str(k): v for k, v in snapshots.items()},
        "final": final,
        "revivals": revivals,
        "cluster_freqs": task.freqs.tolist(),
        "seconds": time.time() - t0,
    }
