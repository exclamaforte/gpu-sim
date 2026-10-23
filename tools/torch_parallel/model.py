"""A small transformer LM, sized to train in seconds on one real GPU.

Deliberately plain: the point of the harness is the parallelism configuration and
the collective traffic, not the model. Keeping it dependency-free (no tokenizer,
no dataset) means a run is reproducible and fast enough to sweep.

Parameters are computed rather than hard-coded so the harness can report the
model shape that produced a trace.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.distributed as dist


@dataclass
class ModelSpec:
    vocab: int = 4096
    seq: int = 512
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 1024
    dropout: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    def param_count(self) -> int:
        """Analytic parameter count, used for memory reporting."""
        per_layer = 4 * self.d_model ** 2 + 2 * self.d_model * self.d_ff
        return self.vocab * self.d_model + self.n_layers * per_layer + self.vocab * self.d_model


class Block(nn.Module):
    def __init__(self, spec: ModelSpec):
        super().__init__()
        d = spec.d_model
        self.n_heads = spec.n_heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.ff1 = nn.Linear(d, spec.d_ff, bias=False)
        self.ff2 = nn.Linear(spec.d_ff, d, bias=False)
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.drop = nn.Dropout(spec.dropout)

    def forward(self, x):
        b, s, d = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h).view(b, s, 3, self.n_heads, d // self.n_heads)
        q, k, v = qkv.unbind(dim=2)
        att = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        )
        att = att.transpose(1, 2).reshape(b, s, d)
        x = x + self.drop(self.proj(att))
        h = self.ln2(x)
        x = x + self.drop(self.ff2(torch.nn.functional.gelu(self.ff1(h))))
        return x


class TinyLM(nn.Module):
    def __init__(self, spec: ModelSpec):
        super().__init__()
        self.spec = spec
        self.embed = nn.Embedding(spec.vocab, spec.d_model)
        self.blocks = nn.ModuleList(Block(spec) for _ in range(spec.n_layers))
        self.lnf = nn.LayerNorm(spec.d_model)
        self.head = nn.Linear(spec.d_model, spec.vocab, bias=False)
        self.pos = nn.Parameter(torch.zeros(1, spec.seq, spec.d_model))
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, idx):
        x = self.embed(idx) + self.pos[:, : idx.shape[1]]
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.lnf(x))

    def loss(self, idx):
        """Next-token cross entropy on a deterministic synthetic batch."""
        logits = self(idx[:, :-1])
        target = idx[:, 1:]
        return torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), target.reshape(-1)
        )


# --------------------------------------------------------------------------
# Tensor parallelism
# --------------------------------------------------------------------------
#
# Megatron-style sharding: a column-parallel layer splits OUTPUT features, so its
# output is already sharded and needs no communication; a row-parallel layer
# splits INPUT features, so its partial sums must be reduced and that reduction is
# the only collective in the forward pass. Stacking them inside a block
# (col -> row) means one all-reduce per sub-layer instead of one per matmul.
#
# The interposer simulates timing, not payload, so an all_reduce here does not
# actually sum anything: the VALUES are wrong and the TRAFFIC is right. That is
# the same bargain data-parallel mode makes, stated here because TP makes it
# easier to forget -- the loss curve in TP mode is not meaningful.

class _ReduceSum(torch.autograd.Function):
    """All-reduce as an autograd node: sum forward, sum of gradients backward.

    A bare `dist.all_reduce` inside forward is invisible to autograd, so the
    backward pass issues no collective at all -- measured: 8 all-reduces per step
    under tp=2, which is forward-only (2 per layer). Megatron's schedule has two
    per sub-layer, and the backward one carries the same bytes, so omitting it
    understates tensor-parallel traffic by roughly half. This makes both passes
    issue the collective, which is what the trace is supposed to show.

    NOTE the arithmetic is still not right (the interposer simulates timing, not
    payload), so the backward reduction is a timing event, not a correct sum.
    """

    @staticmethod
    def forward(ctx, x, group):
        y = x.clone()
        if group is not None:
            dist.all_reduce(y, group=group)
        ctx.group = group
        return y

    @staticmethod
    def backward(ctx, grad):
        g = grad.clone()
        if ctx.group is not None:
            dist.all_reduce(g, group=ctx.group)
        return g, None


class _Col(nn.Module):
    """Column-parallel: weight [out/tp, in]; output sharded on the last dim."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):
        return x @ self.weight.t()


class _Row(nn.Module):
    """Row-parallel: weight [out, in/tp]; input sharded, output reduced."""

    def __init__(self, in_features: int, out_features: int, group):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.normal_(self.weight, std=0.02)
        self.group = group

    def forward(self, x):
        return _ReduceSum.apply(x @ self.weight.t(), self.group)


class BlockTP(nn.Module):
    def __init__(self, spec: ModelSpec, tp_size: int, group):
        super().__init__()
        d, d_ff = spec.d_model, spec.d_ff
        self.n_heads = spec.n_heads // tp_size
        self.head_dim = d // spec.n_heads
        self.qkv = _Col(d, 3 * d // tp_size)
        self.proj = _Row(d // tp_size, d, group)
        self.ff1 = _Col(d, d_ff // tp_size)
        self.ff2 = _Row(d_ff // tp_size, d, group)
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)

    def forward(self, x):
        b, s, d = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h).view(b, s, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        att = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        )
        att = att.transpose(1, 2).reshape(b, s, self.n_heads * self.head_dim)
        x = x + self.proj(att)
        h = self.ln2(x)
        x = x + self.ff2(torch.nn.functional.gelu(self.ff1(h)))
        return x


class TinyLMTP(nn.Module):
    """TinyLM with MLP and attention projections sharded across `tp_size` ranks.

    Embedding, final norm and head stay replicated: vocabulary-parallel output is
    a separate axis, and adding it here would mix two effects before either is
    understood.
    """

    def __init__(self, spec: ModelSpec, tp_size: int = 1, group=None):
        super().__init__()
        self.spec = spec
        self.tp_size = tp_size
        self.embed = nn.Embedding(spec.vocab, spec.d_model)
        self.blocks = nn.ModuleList(BlockTP(spec, tp_size, group)
                                    for _ in range(spec.n_layers))
        self.lnf = nn.LayerNorm(spec.d_model)
        self.head = nn.Linear(spec.d_model, spec.vocab, bias=False)
        self.pos = nn.Parameter(torch.zeros(1, spec.seq, spec.d_model))
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, idx):
        x = self.embed(idx) + self.pos[:, : idx.shape[1]]
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.lnf(x))

    def loss(self, idx):
        logits = self(idx[:, :-1])
        return torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), idx[:, 1:].reshape(-1)
        )

    def local_param_count(self) -> int:
        """Parameters actually resident on this rank (the memory TP buys back)."""
        return sum(p.numel() for p in self.parameters())
