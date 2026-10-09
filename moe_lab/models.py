from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn as nn
import torch.nn.functional as F

from .moe import MoEConfig, MoELayer


class SyntheticMoE(nn.Module):
    """A single MoE layer mapping x -> y. A "dense" baseline is the same thing with one expert."""

    def __init__(self, cfg: MoEConfig):
        super().__init__()
        self.moe = MoELayer(cfg)

    def forward(self, x, labels=None):
        return self.moe(x, labels=labels)

    def moe_layers(self):
        return [self.moe]


class DeepSyntheticMoE(nn.Module):
    """Residual stack of MoE layers on the synthetic task: in-proj -> L x [h + MoE(LN(h))] -> LN -> out-proj.

    Unlike SyntheticMoE, every router sees *learned* representations, so the network can move tokens toward a
    favoured expert (the representation pathway), and tasks that are not linearly separable in x can still be
    routed by task in later layers. A dense baseline is the same model with one expert per layer.
    """

    def __init__(self, cfg: MoEConfig, d_in: int, d_out: int, n_layers: int = 2, d_model: int = 32,
                 pre_dense: int = 0):
        super().__init__()
        self.inp = nn.Linear(d_in, d_model)
        # Optional dense residual MLP before the first MoE layer: gives the first router nonlinear features, the
        # way attention / earlier layers do in a transformer.
        self.pre = None
        if pre_dense:
            self.pre = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, pre_dense), nn.GELU(),
                                     nn.Linear(pre_dense, d_model))
        self.norms = nn.ModuleList(nn.LayerNorm(d_model) for _ in range(n_layers))
        self.moes = nn.ModuleList(MoELayer(replace(cfg, d_in=d_model, d_out=d_model)) for _ in range(n_layers))
        self.ln_f = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, d_out)

    def forward(self, x, labels=None):
        h = self.inp(x)
        if self.pre is not None:
            h = h + self.pre(h)
        for ln, moe in zip(self.norms, self.moes):
            h = h + moe(ln(h), labels=labels)
        return self.out(self.ln_f(h))

    def moe_layers(self):
        return list(self.moes)


class CausalSelfAttention(nn.Module):
    def __init__(self, d: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.n_heads, D // self.n_heads).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(y.transpose(1, 2).reshape(B, T, D))


class Block(nn.Module):
    def __init__(self, d: int, n_heads: int, moe_cfg: MoEConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = CausalSelfAttention(d, n_heads)
        self.ln2 = nn.LayerNorm(d)
        self.moe = MoELayer(moe_cfg)

    def forward(self, x, token_ids, group_ids):
        x = x + self.attn(self.ln1(x))
        B, T, D = x.shape
        h = self.moe(self.ln2(x).reshape(B * T, D), token_ids=token_ids.reshape(-1), group_ids=group_ids)
        return x + h.reshape(B, T, D)


class TinyMoETransformer(nn.Module):
    """Byte-level decoder-only transformer whose FFNs are MoE layers."""

    def __init__(self, moe_cfg: MoEConfig, n_layers: int = 2, d_model: int = 128, n_heads: int = 4,
                 seq_len: int = 128, vocab: int = 256):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(seq_len, d_model)
        self.blocks = nn.ModuleList(Block(d_model, n_heads, moe_cfg) for _ in range(n_layers))
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)

    def forward(self, idx):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        group_ids = torch.arange(B, device=idx.device).repeat_interleave(T)  # one group per sequence
        for blk in self.blocks:
            x = blk(x, idx, group_ids)
        return self.head(self.ln_f(x))

    def moe_layers(self):
        return [b.moe for b in self.blocks]
