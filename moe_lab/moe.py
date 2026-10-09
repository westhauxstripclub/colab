"""Mixture-of-Experts layer with pluggable routing and load-balancing strategies.

Every router variant is reduced to the same two tensors, which keeps comparisons apples-to-apples:

    M[N, E]  bool   dispatch mask: which experts process which token
    G[N, E]  float  combine weights: how much each expert's output counts (zero where M is False)

    y[n] = sum_e G[n, e] * expert_e(x[n])

Gradient facts that matter for the collapse story (all visible in `_route`):
  * An expert with G[n, e] == 0 receives *zero* gradient from token n. Unchosen experts do not learn.
  * softmax_topk (Switch / GShard style): G = softmax(logits) masked to the top-k. The router gets a
    task gradient on every logit through the softmax denominator.
  * topk_renorm (Mixtral style): G = softmax over the selected logits only. Unselected logits get no
    task gradient at all, and with k == 1 the gate is identically 1, so the router gets *no* task
    gradient whatsoever (it can only learn from auxiliary losses).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

TOKEN_CHOICE_ROUTERS = ("softmax_topk", "topk_renorm", "sigmoid_topk", "sinkhorn")
ROUTERS = TOKEN_CHOICE_ROUTERS + ("expert_choice", "hash", "oracle")


@dataclass
class MoEConfig:
    d_in: int
    d_out: int
    d_hidden: int  # hidden width of each expert MLP
    n_experts: int = 8
    top_k: int = 1
    router: str = "softmax_topk"  # one of ROUTERS
    router_type: str = "linear"  # "linear" | "cosine" (X-MoE style, bounded logits)
    router_init_std: float = 0.02
    cosine_dim: int = 8
    cosine_scale: float = 10.0
    capacity_factor: float | None = None  # None = no token dropping (token-choice routers)
    noise_std: float = 0.0  # exploration noise added to selection scores during training
    aux_loss_coef: float = 0.0  # Switch-Transformer load-balancing loss
    aux_scope: str = "batch"  # "batch" (global) | "sequence" (per group_id, i.e. micro-batch level)
    z_loss_coef: float = 0.0  # ST-MoE router z-loss
    bias_update_rate: float = 0.0  # >0 enables DeepSeek-V3 auxiliary-loss-free bias balancing
    identical_init: bool = False  # all experts start as exact copies
    zero_init_output: bool = False  # experts start by outputting exactly 0 (no "noise" for the router to react to)
    compute_all: bool = True  # run every expert on every token (exact same gradients; enables diagnostics)
    hash_vocab: int = 256  # vocabulary size for token-hash routing
    sinkhorn_iters: int = 3


class Experts(nn.Module):
    """E independent 2-layer GELU MLPs stored as batched tensors."""

    def __init__(self, n_experts: int, d_in: int, d_hidden: int, d_out: int):
        super().__init__()
        self.w1 = nn.Parameter(torch.empty(n_experts, d_in, d_hidden))
        self.b1 = nn.Parameter(torch.zeros(n_experts, d_hidden))
        self.w2 = nn.Parameter(torch.empty(n_experts, d_hidden, d_out))
        self.b2 = nn.Parameter(torch.zeros(n_experts, d_out))
        nn.init.uniform_(self.w1, -1 / math.sqrt(d_in), 1 / math.sqrt(d_in))
        nn.init.uniform_(self.w2, -1 / math.sqrt(d_hidden), 1 / math.sqrt(d_hidden))

    def params(self):
        return (self.w1, self.b1, self.w2, self.b2)

    def forward_all(self, x: torch.Tensor) -> torch.Tensor:
        """[N, d_in] -> [N, E, d_out]: every expert applied to every token."""
        h = F.gelu(torch.einsum("nd,edh->neh", x, self.w1) + self.b1)
        return torch.einsum("neh,eho->neo", h, self.w2) + self.b2

    def forward_one(self, x: torch.Tensor, e: int) -> torch.Tensor:
        h = F.gelu(x @ self.w1[e] + self.b1[e])
        return h @ self.w2[e] + self.b2[e]


class Router(nn.Module):
    def __init__(self, cfg: MoEConfig):
        super().__init__()
        self.cfg = cfg
        E = cfg.n_experts
        if cfg.router_type == "linear":
            self.weight = nn.Parameter(torch.randn(E, cfg.d_in) * cfg.router_init_std)
        elif cfg.router_type == "cosine":
            # X-MoE (Chi et al. 2022): route in a low-dimensional space on the unit sphere with a fixed
            # scale, so logits cannot grow without bound.
            self.proj = nn.Linear(cfg.d_in, cfg.cosine_dim, bias=False)
            self.weight = nn.Parameter(torch.randn(E, cfg.cosine_dim) * cfg.router_init_std)
        else:
            raise ValueError(cfg.router_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.cfg.router_type == "linear":
            return x @ self.weight.t()
        z = F.normalize(self.proj(x), dim=-1)
        w = F.normalize(self.weight, dim=-1)
        return self.cfg.cosine_scale * (z @ w.t())


def sinkhorn_scores(logits: torch.Tensor, n_iters: int) -> torch.Tensor:
    """Log-domain Sinkhorn: rows sum to 1, columns to N/E. Used only to *select* experts (S-BASE)."""
    N, E = logits.shape
    log_p = logits.float()
    for _ in range(n_iters):
        log_p = log_p - log_p.logsumexp(dim=1, keepdim=True)
        log_p = log_p - log_p.logsumexp(dim=0, keepdim=True) + math.log(N / E)
    return log_p


class MoELayer(nn.Module):
    def __init__(self, cfg: MoEConfig):
        super().__init__()
        assert cfg.router in ROUTERS, cfg.router
        self.cfg = cfg
        self.router = Router(cfg)
        self.experts = Experts(cfg.n_experts, cfg.d_in, cfg.d_hidden, cfg.d_out)
        if cfg.zero_init_output:
            with torch.no_grad():
                self.experts.w2.zero_()
        if cfg.identical_init:
            with torch.no_grad():
                for p in self.experts.params():
                    p.copy_(p[:1].clone().expand_as(p))
        self.register_buffer("bias", torch.zeros(cfg.n_experts))  # aux-loss-free balancing bias
        self.register_buffer("hash_proj", torch.randn(cfg.d_in, cfg.n_experts))
        g = torch.Generator().manual_seed(1234)
        self.register_buffer("hash_table", torch.randperm(cfg.hash_vocab, generator=g) % cfg.n_experts)
        self.force_expert: int | None = None  # used for "head start" pre-training of one expert
        self.disabled: list[int] = []  # experts removed from routing (ablation analysis)
        self.capture_input = False
        self.bias_frozen = False  # keep applying the bias but stop updating it
        self.last: dict = {}
        self.aux_losses: dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------ routing
    def _route(self, x, logits, token_ids=None, labels=None):
        cfg, E, k = self.cfg, self.cfg.n_experts, self.cfg.top_k
        N = x.shape[0]
        probs = logits.softmax(-1)
        dev = x.device

        def one_hot_mask(idx):
            return torch.zeros(N, E, dtype=torch.bool, device=dev).scatter_(1, idx, True)

        if self.force_expert is not None:
            M = one_hot_mask(torch.full((N, 1), self.force_expert, device=dev))
            return M, M.float(), M, probs

        r = cfg.router
        if r in TOKEN_CHOICE_ROUTERS:
            scores = logits.detach()
            if r == "sinkhorn":
                scores = sinkhorn_scores(scores, cfg.sinkhorn_iters)
            elif r == "sigmoid_topk":
                scores = torch.sigmoid(scores)
            if cfg.bias_update_rate > 0:
                scores = scores + self.bias  # bias affects selection only, never the gate value
            if self.training and cfg.noise_std > 0:
                scores = scores + cfg.noise_std * torch.randn_like(scores)
            if self.disabled:
                scores = scores.clone()
                scores[:, self.disabled] = float("-inf")
            M = one_hot_mask(scores.topk(k, dim=-1).indices)
            if r in ("softmax_topk", "sinkhorn"):
                G = probs * M
            elif r == "topk_renorm":
                G = logits.masked_fill(~M, float("-inf")).softmax(-1)
            else:  # sigmoid_topk (DeepSeek-V3): normalise sigmoid scores over the selected experts
                s = torch.sigmoid(logits) * M
                G = s / s.sum(-1, keepdim=True).clamp_min(1e-9)
            M_sel = M
            if cfg.capacity_factor is not None:
                C = math.ceil(cfg.capacity_factor * N * k / E)
                prio = torch.where(M, probs.detach(), torch.full_like(probs, -1.0))
                rank = prio.argsort(0, descending=True).argsort(0)  # per-expert priority rank
                keep = rank < C
                M, G = M & keep, G * keep
            return M, G, M_sel, probs

        if r == "expert_choice":
            # Zhou et al. 2022: each expert picks its top-C tokens. Balanced by construction, but a token
            # may get 0 or many experts, and it is non-causal in autoregressive LMs (sees the whole batch).
            cf = cfg.capacity_factor or 1.0
            C = max(1, min(N, int(round(cf * N * k / E))))
            top = probs.detach().t().topk(C, dim=-1).indices  # [E, C]
            M = torch.zeros(E, N, dtype=torch.bool, device=dev).scatter_(1, top, True).t()
            return M, probs * M, M, probs

        if r == "hash":
            # Roller et al. 2021: fixed routing with no learned parameters. Token-id hash when ids are
            # available, otherwise a fixed random projection of the input.
            if token_ids is not None:
                idx = self.hash_table[token_ids.long()].unsqueeze(-1)
                if k > 1:
                    idx = torch.cat([idx] + [(idx + j) % E for j in range(1, k)], dim=-1)
            else:
                idx = (x.detach() @ self.hash_proj).topk(k, dim=-1).indices
            M = one_hot_mask(idx)
            return M, M.float() / k, M, probs

        if r == "oracle":  # ground-truth routing: label -> expert (upper bound on what specialisation buys)
            M = one_hot_mask((labels.long() % E).unsqueeze(-1))
            return M, M.float(), M, probs

        raise ValueError(r)

    # ------------------------------------------------------------------ losses
    def _switch_aux(self, M_sel, P, group_ids):
        """Switch Transformer loss E * sum_e f_e * P_e (== 1 at perfect balance)."""
        E, k = self.cfg.n_experts, self.cfg.top_k
        Mf = M_sel.float()
        if self.cfg.aux_scope == "sequence" and group_ids is not None:
            n_groups = int(group_ids.max().item()) + 1
            onehot = F.one_hot(group_ids.long(), n_groups).float()  # [N, G]
            counts = onehot.sum(0).clamp_min(1).unsqueeze(-1)
            f = onehot.t() @ Mf / (counts * k)
            Pg = onehot.t() @ P / counts
            return E * (f * Pg).sum(-1).mean()
        f = Mf.mean(0) / k
        return E * (f * P.mean(0)).sum()

    # ------------------------------------------------------------------ forward
    def forward(self, x, *, token_ids=None, group_ids=None, labels=None):
        cfg = self.cfg
        logits = self.router(x)
        M, G, M_sel, probs = self._route(x, logits, token_ids, labels)
        if self.disabled and cfg.router not in TOKEN_CHOICE_ROUTERS:  # fixed routers: just drop the expert
            M, G = M.clone(), G.clone()
            M[:, self.disabled] = False
            G[:, self.disabled] = 0.0

        self.aux_losses = {}
        learned_token_choice = cfg.router in TOKEN_CHOICE_ROUTERS and self.force_expert is None
        if cfg.aux_loss_coef > 0 and learned_token_choice:
            P = probs
            if cfg.router == "sigmoid_topk":
                s = torch.sigmoid(logits)
                P = s / s.sum(-1, keepdim=True)
            self.aux_losses["aux"] = cfg.aux_loss_coef * self._switch_aux(M_sel, P, group_ids)
        if cfg.z_loss_coef > 0:
            self.aux_losses["z"] = cfg.z_loss_coef * logits.logsumexp(-1).pow(2).mean()

        if cfg.compute_all:
            out = torch.einsum("ne,neo->no", G, self.experts.forward_all(x))
        else:
            out = x.new_zeros(x.shape[0], cfg.d_out)
            for e in range(cfg.n_experts):
                idx = M[:, e].nonzero(as_tuple=True)[0]
                if idx.numel():
                    y = self.experts.forward_one(x[idx], e)
                    out = out.index_add(0, idx, G[idx, e : e + 1] * y)

        self.last = {
            "logits": logits.detach(),
            "probs": probs.detach(),
            "M": M.detach(),
            "M_sel": M_sel.detach(),
            "G": G.detach(),
        }
        if self.capture_input:
            self.last["x"] = x.detach()
        return out

    def aux_loss(self) -> torch.Tensor | float:
        return sum(self.aux_losses.values()) if self.aux_losses else 0.0

    # ------------------------------------------------------------------ balancing / diagnostics
    @torch.no_grad()
    def update_bias(self):
        """DeepSeek-V3 aux-loss-free balancing: nudge under-loaded experts up, over-loaded down."""
        if self.cfg.bias_update_rate <= 0 or self.bias_frozen or "M_sel" not in self.last:
            return
        load = self.last["M_sel"].float().sum(0)
        self.bias += self.cfg.bias_update_rate * torch.sign(load.mean() - load)

    def solo_gate(self, probs: torch.Tensor) -> torch.Tensor:
        """Gate an expert would receive if it were the only one chosen (for counterfactual diagnostics)."""
        if self.cfg.router in ("softmax_topk", "sinkhorn", "expert_choice"):
            return probs
        return torch.ones_like(probs)

    @torch.no_grad()
    def counterfactual_outputs(self, x: torch.Tensor) -> torch.Tensor:
        """[N, E, d_out]: what each expert would output for each token if routed there alone."""
        probs = self.router(x).softmax(-1)
        return self.solo_gate(probs).unsqueeze(-1) * self.experts.forward_all(x)
