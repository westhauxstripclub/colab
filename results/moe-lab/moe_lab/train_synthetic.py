"""Training + diagnostics on the synthetic mixture-of-tasks problem (ground-truth task ids known).

depth = 0 trains the original single MoE layer on fixed inputs (SyntheticMoE). depth >= 1 trains a residual stack of
MoE layers on learned representations (DeepSyntheticMoE). Single-layer runs report every diagnostic at the top
level, exactly as before; deep runs report per-layer diagnostics under "layers" and layer means at the top level.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from .data import make_cluster_task
from .metrics import (entropy_bits, expert_redundancy, gini, load_stats, logit_decomposition, routed_cka,
                      router_stats, specialization, spearman, substitution_gap)
from .models import DeepSyntheticMoE, SyntheticMoE
from .moe import TOKEN_CHOICE_ROUTERS, MoEConfig

SYNTH_MOE_DEFAULTS = dict(d_hidden=16, n_experts=8, top_k=1, router="softmax_topk")
DEFAULT_LR = {"adam": 3e-3, "sgd": 0.05}


@dataclass
class SynthRunConfig:
    name: str
    seed: int = 0
    moe: dict = field(default_factory=dict)  # overrides for MoEConfig
    task: dict = field(default_factory=dict)  # overrides for make_cluster_task
    depth: int = 0  # 0: single MoE layer on raw inputs; >=1: DeepSyntheticMoE with this many MoE layers
    d_model: int = 32  # residual width of the deep model
    pre_dense: int = 0  # >0: dense residual MLP of this width before the first MoE layer (deep model only)
    steps: int = 3000
    batch_size: int = 256
    lr: float = 3e-3  # learning rate of `optimizer`
    optimizer: str = "adam"  # "adam" | "sgd": everything except (optionally) the router
    router_optimizer: str | None = None  # None = same as `optimizer`; otherwise router gets its own optimizer
    sgd_momentum: float = 0.9
    router_lr_mult: float = 1.0  # router learning rate relative to its optimizer's base learning rate
    router_grad_rownorm: bool = False  # give every expert's router row a gradient of equal norm (SGD probe)
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
    decomp_probe: int = 1024  # probe tokens for the representation-vs-router logit decomposition (0 = off)


def _per_token_loss(pred, y, kind):
    if kind == "mse":
        return (pred - y).pow(2).mean(-1)
    return F.cross_entropy(pred, y, reduction="none")


def _loss(pred, y, kind):
    return F.mse_loss(pred, y) if kind == "mse" else F.cross_entropy(pred, y)


def _layer_stats(layer, c):
    st = load_stats(layer.last["M"])
    st.update(router_stats(layer.last["logits"]))
    st.update(specialization(layer.last["G"].argmax(-1).numpy(), c.numpy()))
    if layer.cfg.bias_update_rate > 0:
        st["bias"] = layer.bias.tolist()
    return st


def _summarise(per_layer):
    """Top level = the layer itself for single-layer models, the mean over layers otherwise."""
    if len(per_layer) == 1:
        return dict(per_layer[0])
    keys = [k for k, v in per_layer[0].items() if isinstance(v, (int, float))]
    out = {k: float(np.mean([l[k] for l in per_layer])) for k in keys}
    out["layers"] = per_layer
    return out


@torch.no_grad()
def light_eval(model, task, step):
    model.eval()
    pred = model(task.x_val, labels=task.c_val)
    st = {"step": step, "val_loss": _loss(pred, task.y_val, task.loss).item()}
    if task.loss == "ce":
        st["val_acc"] = (pred.argmax(-1) == task.y_val).float().mean().item()
    st.update(_summarise([_layer_stats(l, task.c_val) for l in model.moe_layers()]))
    n = min(len(task.x_val), len(task.x_train))
    st["train_loss"] = _loss(model(task.x_train[:n], labels=task.c_train[:n]), task.y_train[:n], task.loss).item()
    model.train()
    return st


@torch.no_grad()
def capture_router_inputs(model, x, labels):
    """Run x through the model and return the input each MoE layer's router saw."""
    model.eval()
    layers = model.moe_layers()
    for l in layers:
        l.capture_input = True
    model(x, labels=labels)
    xs = [l.last["x"].clone() for l in layers]
    for l in layers:
        l.capture_input = False
        l.last.pop("x", None)
    model.train()
    return xs


@torch.no_grad()
def heavy_eval(model, task):
    """Counterfactual diagnostics (single-layer models only): what would each expert do on each task?"""
    if len(model.moe_layers()) != 1 or not isinstance(model, SyntheticMoE):
        return {}
    model.eval()
    layer = model.moe
    E, C = layer.cfg.n_experts, task.n_clusters
    x, y, c = task.x_val, task.y_val, task.c_val
    pred = model(x, labels=c)
    tok_loss = _per_token_loss(pred, y, task.loss)
    M, G = layer.last["M"], layer.last["G"]
    top1 = G.argmax(-1)

    cf = layer.counterfactual_outputs(x)  # [N, E, d]
    if task.loss == "mse":
        cf_loss = (cf - y.unsqueeze(1)).pow(2).mean(-1)  # [N, E]
    else:
        cf_loss = F.cross_entropy(cf.reshape(-1, cf.shape[-1]), y.repeat_interleave(E), reduction="none").view(-1, E)
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
        "routed_cka": routed_cka(cf, top1, alive),
        "substitution_gap": substitution_gap(cf_loss, top1, alive),
    }
    raw = layer.experts.forward_all(x[:2000])
    out.update({f"all_{k}": v for k, v in expert_redundancy(raw).items()})
    out.update({f"alive_{k}": v for k, v in expert_redundancy(raw, alive).items()})
    model.train()
    return out


@torch.no_grad()
def overfitting_eval(model, task):
    """Is a heavily used expert *overfit*? Compare train vs val loss on the tokens routed to each expert."""
    if not isinstance(model, SyntheticMoE):
        return {}
    model.eval()
    layer = model.moe
    E = layer.cfg.n_experts
    res = {}
    for split, x, y, c in (("train", task.x_train, task.y_train, task.c_train), ("val", task.x_val, task.y_val, task.c_val)):
        pred = model(x, labels=c)
        tl = _per_token_loss(pred, y, task.loss)
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
    """Loss increase when each expert is removed from routing (tokens fall back to the next choice), per layer."""
    model.eval()
    base = _loss(model(task.x_val, labels=task.c_val), task.y_val, task.loss).item()
    per_layer = []
    for layer in model.moe_layers():
        deltas = []
        for e in range(layer.cfg.n_experts):
            layer.disabled = [e]
            deltas.append(_loss(model(task.x_val, labels=task.c_val), task.y_val, task.loss).item() - base)
        layer.disabled = []
        share = np.clip(deltas, 0, None)
        per_layer.append({
            "ablation_delta": deltas,
            "ablation_gini": gini(share),
            "useful_experts": float(2 ** entropy_bits(share)) if share.sum() > 0 else 0.0,
        })
    model.train()
    if len(per_layer) == 1:
        return per_layer[0]
    return {"useful_experts": float(np.mean([p["useful_experts"] for p in per_layer])), "ablation_layers": per_layer}


@torch.no_grad()
def test_eval(model, task):
    model.eval()
    pred = model(task.x_test, labels=task.c_test)
    out = {"test_loss": _loss(pred, task.y_test, task.loss).item()}
    if task.loss == "ce":
        out["test_acc"] = (pred.argmax(-1) == task.y_test).float().mean().item()
    out["per_cluster_test_loss"] = [
        _per_token_loss(pred[task.c_test == k], task.y_test[task.c_test == k], task.loss).mean().item()
        for k in range(task.n_clusters)
    ]
    model.train()
    return out


def _scale_expert_grads(layer):
    load = layer.last["M"].float().sum(0)
    scale = load.mean() / load.clamp_min(1.0)
    for p in layer.experts.params():
        if p.grad is not None:
            p.grad.mul_(scale.view(-1, *[1] * (p.dim() - 1)))


def _rownorm_router_grads(layer):
    w = layer.router.weight
    if w.grad is None:
        return
    norms = w.grad.norm(dim=1, keepdim=True)
    nz = norms > 0
    if nz.any():
        target = norms[nz].mean()
        w.grad.copy_(torch.where(nz, w.grad * target / norms.clamp_min(1e-12), w.grad))


@torch.no_grad()
def _revive_dead(layer, opts, ema_load, threshold):
    E = layer.cfg.n_experts
    revived = []
    for d in (ema_load < threshold / E).nonzero(as_tuple=True)[0].tolist():
        b = int(ema_load.argmax())
        tensors = list(layer.experts.params())
        if layer.cfg.router_type == "linear":
            tensors.append(layer.router.weight)
        for p in tensors:
            p[d] = p[b] + 0.01 * p[b].std() * torch.randn_like(p[b])
            for opt in opts:
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


def _make_opt(kind, groups, momentum):
    return torch.optim.Adam(groups) if kind == "adam" else torch.optim.SGD(groups, lr=groups[0]["lr"], momentum=momentum)


def run_synthetic(rc: SynthRunConfig) -> dict:
    t0 = time.time()
    torch.manual_seed(rc.seed)
    task = make_cluster_task(seed=rc.seed, **rc.task)
    d_in = task.x_train.shape[1]
    d_out = task.y_train.shape[1] if task.loss == "mse" else int(rc.task.get("d_out", 8))
    cfg = MoEConfig(d_in=d_in, d_out=d_out, **{**SYNTH_MOE_DEFAULTS, **rc.moe})
    if rc.depth == 0:
        model = SyntheticMoE(cfg)
    else:
        model = DeepSyntheticMoE(cfg, d_in, d_out, n_layers=rc.depth, d_model=rc.d_model, pre_dense=rc.pre_dense)
    layers = model.moe_layers()
    if rc.freeze_router:
        for l in layers:
            l.router.requires_grad_(False)
    if rc.freeze_experts:
        for l in layers:
            l.experts.requires_grad_(False)

    router_params = [p for l in layers for p in l.router.parameters() if p.requires_grad]
    router_ids = {id(p) for p in router_params}
    main_params = [p for p in model.parameters() if p.requires_grad and id(p) not in router_ids]
    router_kind = rc.router_optimizer or rc.optimizer
    main_lr = rc.lr
    router_lr = (rc.lr if router_kind == rc.optimizer else DEFAULT_LR[router_kind]) * rc.router_lr_mult
    if router_kind == rc.optimizer:  # one optimizer, two groups (identical to the original implementation)
        groups = [g for g in ({"params": main_params, "lr": main_lr}, {"params": router_params, "lr": router_lr}) if g["params"]]
        opts = [_make_opt(rc.optimizer, groups, rc.sgd_momentum)]
    else:
        opts = [_make_opt(rc.optimizer, [{"params": main_params, "lr": main_lr}], rc.sgd_momentum)]
        if router_params:
            opts.append(_make_opt(router_kind, [{"params": router_params, "lr": router_lr}], rc.sgd_momentum))
    gen = torch.Generator().manual_seed(rc.seed)
    n_train = len(task.x_train)

    def train_step():
        idx = torch.randint(0, n_train, (rc.batch_size,), generator=gen)
        x, y, c = task.x_train[idx], task.y_train[idx], task.c_train[idx]
        loss = _loss(model(x, labels=c), y, task.loss) + sum(l.aux_loss() for l in layers)
        for opt in opts:
            opt.zero_grad(set_to_none=True)
        loss.backward()
        if layers[0].force_expert is None and train_step.n < rc.router_warmup_steps:
            for p in router_params:
                p.grad = None
        train_step.n += 1
        for l in layers:
            if rc.load_normalized_grads:
                _scale_expert_grads(l)
            if rc.router_grad_rownorm and l.cfg.router_type == "linear":
                _rownorm_router_grads(l)
        for opt in opts:
            opt.step()
        for l in layers:
            l.update_bias()

    train_step.n = 0
    if rc.head_start_steps:
        for l in layers:
            l.force_expert = 0
        for _ in range(rc.head_start_steps):
            train_step()
        for l in layers:
            l.force_expert = None

    probe_x = task.x_val[: rc.decomp_probe]
    probe_c = task.c_val[: rc.decomp_probe]
    x0s = capture_router_inputs(model, probe_x, probe_c) if rc.decomp_probe else None
    params0 = [{k: v.detach().clone() for k, v in l.router.named_parameters()} for l in layers]

    history, snapshots, revivals = [], {}, []
    ema_loads = [torch.full((cfg.n_experts,), 1.0 / cfg.n_experts) for _ in layers]
    for step in range(rc.steps + 1):
        if step % rc.eval_every == 0 or step == rc.steps or (step < 300 and step % 10 == 0):
            st = light_eval(model, task, step)
            if rc.decomp_probe:
                xts = capture_router_inputs(model, probe_x, probe_c)
                st["decomp"] = [logit_decomposition(l, x0, xt, p0, l.cfg.top_k)
                                for l, x0, xt, p0 in zip(layers, x0s, xts, params0)]
            history.append(st)
        if step in rc.snapshot_steps:
            snapshots[step] = heavy_eval(model, task)
        if step == rc.steps:
            break
        if rc.balance_until and step == rc.balance_until:
            for l in layers:
                l.cfg.aux_loss_coef = 0.0
                l.bias_frozen = True
                if l.cfg.router == "sinkhorn":
                    l.cfg.router = "softmax_topk"
        train_step()
        if rc.revive_every and cfg.router in TOKEN_CHOICE_ROUTERS:
            for li, (l, ema) in enumerate(zip(layers, ema_loads)):
                frac = l.last["M"].float().sum(0)
                ema.mul_(0.95).add_(0.05 * frac / frac.sum())
                if (step + 1) % rc.revive_every == 0 and step + 1 < 0.8 * rc.steps:
                    for d in _revive_dead(l, opts, ema, rc.revive_threshold):
                        revivals.append((step + 1, li, d) if len(layers) > 1 else (step + 1, d))

    final = {**history[-1], **heavy_eval(model, task), **overfitting_eval(model, task), **ablation_eval(model, task),
             **test_eval(model, task)}
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
