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
