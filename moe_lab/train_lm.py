"""Training + diagnostics for a tiny byte-level MoE transformer on three text domains."""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from .data import load_text_corpus, sample_text_batch
from .metrics import expert_redundancy, gini, load_stats, mutual_info_bits, router_stats, specialization
from .models import TinyMoETransformer
from .moe import MoEConfig

LM_MOE_DEFAULTS = dict(d_hidden=128, n_experts=8, top_k=2, router="softmax_topk", compute_all=False)


@dataclass
class LMRunConfig:
    name: str
    seed: int = 0
    moe: dict = field(default_factory=dict)
    n_layers: int = 2
    d_model: int = 128
    n_heads: int = 4
    seq_len: int = 128
    batch_size: int = 32
    steps: int = 2000
    lr: float = 2e-3
    warmup: int = 100
    weight_decay: float = 0.01
    eval_every: int = 250
    eval_batches: int = 6
    freeze_router: bool = False
    data_dir: str = "data"
    device: str = "cpu"


def _lr_at(step, rc):
    if step < rc.warmup:
        return rc.lr * (step + 1) / rc.warmup
    t = (step - rc.warmup) / max(1, rc.steps - rc.warmup)
    return rc.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t)))


@torch.no_grad()
def evaluate(model, corpus, rc, full=False):
    model.eval()
    layers = model.moe_layers()
    gen = torch.Generator().manual_seed(999)  # identical eval batches for every run
    n_dom = len(corpus.domains)
    loss_sum, tok_count = np.zeros(n_dom), np.zeros(n_dom)
    per_layer = [{"M": [], "G": [], "logits": [], "x": []} for _ in layers]
    doms, bytes_ = [], []
    for li, layer in enumerate(layers):
        layer.capture_input = full and li >= 0
    for _ in range(rc.eval_batches):
        x, y, d = sample_text_batch(corpus, "val", rc.batch_size, rc.seq_len, gen)
        x, y = x.to(rc.device), y.to(rc.device)
        logits = model(x)
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="none").view(x.shape)
        for k in range(n_dom):
            sel = (d == k).to(rc.device)
            loss_sum[k] += ce[sel].sum().item()
            tok_count[k] += sel.sum().item() * rc.seq_len
        doms.append(d.repeat_interleave(rc.seq_len))
        bytes_.append(x.reshape(-1).cpu())
        for li, layer in enumerate(layers):
            for key in ("M", "G", "logits"):
                per_layer[li][key].append(layer.last[key].cpu())
            if full and len(per_layer[li]["x"]) == 0:
                per_layer[li]["x"].append(layer.last["x"][:2048].cpu())
    for layer in layers:
        layer.capture_input = False
    doms = torch.cat(doms).numpy()
    bytes_ = torch.cat(bytes_).numpy()

    out = {
        "val_loss": float(loss_sum.sum() / tok_count.sum()),
        "val_loss_by_domain": dict(zip(corpus.domains, (loss_sum / np.maximum(tok_count, 1)).tolist())),
        "layers": [],
    }
    for li, layer in enumerate(layers):
        M = torch.cat(per_layer[li]["M"])
        G = torch.cat(per_layer[li]["G"])
        st = load_stats(M)
        st.update(router_stats(torch.cat(per_layer[li]["logits"])))
        if layer.cfg.n_experts > 1:
            top1 = G.argmax(-1).numpy()
            dom_spec = specialization(top1, doms)
            st["domain_info_recovered"] = dom_spec["label_info_recovered"]
            st["routing_explained_by_domain"] = dom_spec["routing_explained"]
            mi_b, h_e, _ = mutual_info_bits(top1, bytes_)
            st["routing_explained_by_byte"] = mi_b / h_e if h_e > 0 else 0.0
            st["domain_expert_load"] = [
                (M[torch.from_numpy(doms == k)].float().sum(0) / max(1, M[torch.from_numpy(doms == k)].sum())).tolist()
                for k in range(len(corpus.domains))
            ]
            if full:
                xs = per_layer[li]["x"][0].to(rc.device)
                raw = layer.experts.forward_all(xs).cpu()
                alive = np.array(st["load"]) >= 0.1 / layer.cfg.n_experts
                st.update({f"alive_{k}": v for k, v in expert_redundancy(raw, alive).items()})
        out["layers"].append(st)
    model.train()
    return out


@torch.no_grad()
def ablation(model, corpus, rc):
    """Per-layer, per-expert increase in val loss when that expert is removed from routing."""
    base = evaluate(model, corpus, rc)["val_loss"]
    res = []
    for layer in model.moe_layers():
        deltas = []
        for e in range(layer.cfg.n_experts):
            layer.disabled = [e]
            deltas.append(evaluate(model, corpus, rc)["val_loss"] - base)
        layer.disabled = []
        share = np.clip(deltas, 0, None)
        res.append({"ablation_delta": deltas, "ablation_gini": gini(share)})
    return res


def run_lm(rc: LMRunConfig) -> dict:
    t0 = time.time()
    torch.manual_seed(rc.seed)
    corpus = load_text_corpus(rc.data_dir)
    cfg = MoEConfig(d_in=rc.d_model, d_out=rc.d_model, **{**LM_MOE_DEFAULTS, **rc.moe})
    model = TinyMoETransformer(cfg, rc.n_layers, rc.d_model, rc.n_heads, rc.seq_len).to(rc.device)
    if rc.freeze_router:
        for layer in model.moe_layers():
            layer.router.requires_grad_(False)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=rc.lr, betas=(0.9, 0.95), weight_decay=rc.weight_decay)
    gen = torch.Generator().manual_seed(rc.seed)
    history = []
    for step in range(rc.steps + 1):
        if step % rc.eval_every == 0 or step == rc.steps:
            ev = evaluate(model, corpus, rc, full=(step == rc.steps))
            ev["step"] = step
            history.append(ev)
        if step == rc.steps:
            break
        for g in opt.param_groups:
            g["lr"] = _lr_at(step, rc)
        x, y, _ = sample_text_batch(corpus, "train", rc.batch_size, rc.seq_len, gen)
        x, y = x.to(rc.device), y.to(rc.device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        loss = loss + sum(layer.aux_loss() for layer in model.moe_layers())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        for layer in model.moe_layers():
            layer.update_bias()
    n_params = sum(p.numel() for p in model.parameters())
    active = n_params - sum(p.numel() for l in model.moe_layers() for p in l.experts.parameters()) + sum(
        p.numel() // cfg.n_experts * cfg.top_k for l in model.moe_layers() for p in l.experts.parameters()
    )
    return {
        "config": asdict(rc),
        "moe_config": asdict(cfg),
        "history": history,
        "final": history[-1],
        "ablation": ablation(model, corpus, rc) if cfg.n_experts > 1 else None,
        "n_params": n_params,
        "n_active_params": active,
        "seconds": time.time() - t0,
    }
