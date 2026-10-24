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
import zlib


def _wgen(seed: int, name: str) -> torch.Generator:
    """Deterministic per-weight generator (gpu-sim-f50).

    crc32 is stable across runs (unlike hash()), so the same (seed, name)
    always draws the same values on every rank. TP ranks draw the FULL
    weight from one generator and slice their shard; PP stages and the
    unsharded reference draw the same names -- agreement by construction,
    which is what the TP/PP gates compare against.
    """
    g = torch.Generator()
    g.manual_seed(zlib.crc32(f"{seed}:{name}".encode()))
    return g


def _draw_full(shape, seed: int, name: str, std: float = 0.02) -> torch.Tensor:
    t = torch.empty(shape)
    nn.init.normal_(t, std=std, generator=_wgen(seed, name))
    return t


def _seed_block(blk, spec: ModelSpec, seed: int, layer_idx: int,
                tp_size: int = 1, tp_rank: int = 0) -> None:
    """Fill one block's weights deterministically (gpu-sim-f50).

    Draws the FULL weight from the per-layer generator and slices the rank's
    shard (rows for Col, cols for Row). Same (seed, layer) on every rank ->
    identical full weights everywhere, so sharded models and the unsharded
    reference agree by construction. Works for Block (tp_size=1, no slice)
    and BlockTP alike; LayerNorms stay at their deterministic init.

    QKV EXCEPTION: the (3, heads, head_dim) view lays rows out q-major
    ([q-all, k-all, v-all]), so a head is NOT a contiguous row slice -- rank
    r owns rows [r*n:(r+1)*n] of EACH third (n = d/tp). Slicing contiguously
    would hand rank 0 all of q plus half of k (measured: 0.13 fwd err).
    """
    d, d_ff = spec.d_model, spec.d_ff
    n = d // tp_size if tp_size > 1 else d
    qkv_groups = ([(j * d + tp_rank * n, j * d + (tp_rank + 1) * n)
                   for j in range(3)] if tp_size > 1 else None)
    for name, shape, sdim, groups in (("qkv", (3 * d, d), 0, qkv_groups),
                                      ("proj", (d, d), 1, None),
                                      ("ff1", (d_ff, d), 0, None),
                                      ("ff2", (d, d_ff), 1, None)):
        full = _draw_full(shape, seed, f"b{layer_idx}.{name}")
        if groups is not None:
            full = torch.cat([full[a:b] for a, b in groups], dim=0)
        elif tp_size > 1:
            m = full.shape[sdim] // tp_size
            sl = [slice(None)] * full.dim()
            sl[sdim] = slice(tp_rank * m, (tp_rank + 1) * m)
            full = full[tuple(sl)]
        with torch.no_grad():
            getattr(blk, name).weight.copy_(full)


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
    def __init__(self, spec: ModelSpec, cp_size: int = 1, cp_rank: int = 0,
                 cp_peers=None):
        super().__init__()
        self.cp_size = cp_size
        self.cp_rank = cp_rank
        self.cp_peers = cp_peers or [0]
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
        hd = d // self.n_heads
        if self.cp_size <= 1:
            # Use the same manual softmax as the CP path so cp=1 and cp>1
            # agree numerically (SDPA's flash backend differs by ~1e-2).
            hd = d // self.n_heads
            qh = q.transpose(1, 2)
            kh = k.transpose(1, 2)
            vh = v.transpose(1, 2)
            scores = (qh @ kh.transpose(-2, -1)) / (hd ** 0.5)
            mask = torch.arange(s, device=x.device)[None, None, None, :] <= torch.arange(s, device=x.device)[None, None, :, None]
            scores = scores.masked_fill(~mask, float("-inf"))
            attn = torch.softmax(scores, dim=-1) @ vh
            att = attn.transpose(1, 2).reshape(b, s, d)
        else:
            import torch.distributed as dist
            cp = self.cp_size
            seq_full = s * cp
            full_k = torch.zeros(b, seq_full, self.n_heads, hd,
                                 device=x.device, dtype=k.dtype)
            full_v = torch.zeros(b, seq_full, self.n_heads, hd,
                                 device=x.device, dtype=v.dtype)
            s = x.shape[1]
            off = self.cp_rank * s
            full_k[:, off:off+s] = k.detach().clone()
            full_v[:, off:off+s] = v.detach().clone()
            for t in range(cp):
                if t == self.cp_rank:
                    continue
                peer = self.cp_peers[t]
                recv_k = torch.empty_like(k)
                recv_v = torch.empty_like(v)
                sk = dist.isend(k.detach().contiguous(), dst=peer)
                rk = dist.irecv(recv_k, src=peer)
                sv = dist.isend(v.detach().contiguous(), dst=peer)
                rv = dist.irecv(recv_v, src=peer)
                sk.wait(); rk.wait(); sv.wait(); rv.wait()
                off_t = t * s
                full_k[:, off_t:off_t+s] = recv_k
                full_v[:, off_t:off_t+s] = recv_v
            qpos = torch.arange(self.cp_rank * s, (self.cp_rank + 1) * s, device=x.device)
            qh = q.transpose(1, 2)
            kh = full_k.permute(0, 2, 1, 3)
            vh = full_v.permute(0, 2, 1, 3)
            scores = (qh @ kh.transpose(-2, -1)) / (hd ** 0.5)
            mask = torch.arange(seq_full, device=x.device)[None, None, None, :] <= qpos[None, None, :, None]
            scores = scores.masked_fill(~mask, float("-inf"))
            attn = torch.softmax(scores, dim=-1) @ vh
            att = attn.transpose(1, 2).reshape(b, s, d)
        x = x + self.drop(self.proj(att))
        h = self.ln2(x)
        x = x + self.drop(self.ff2(torch.nn.functional.gelu(self.ff1(h))))
        return x


class TinyLM(nn.Module):
    def __init__(self, spec: ModelSpec, cp_size: int = 1, cp_rank: int = 0,
                 cp_peers=None):
        super().__init__()
        self.spec = spec
        self.cp_size = cp_size
        self.cp_rank = cp_rank
        self.cp_peers = cp_peers or [0]
        self.embed = nn.Embedding(spec.vocab, spec.d_model)
        self.blocks = nn.ModuleList(
            Block(spec, cp_size=cp_size, cp_rank=cp_rank, cp_peers=cp_peers)
            for _ in range(spec.n_layers))
        self.lnf = nn.LayerNorm(spec.d_model)
        self.head = nn.Linear(spec.d_model, spec.vocab, bias=False)
        self.pos = nn.Parameter(torch.zeros(1, spec.seq, spec.d_model))
        nn.init.normal_(self.pos, std=0.02)

    def seed_weights(self, seed: int) -> None:
        """Deterministically fill ALL params (gpu-sim-f50).

        Same scheme the sharded models use, so the unsharded reference
        agrees with them by construction. LayerNorms keep ones/zeros.
        """
        with torch.no_grad():
            d = self.spec.d_model
            self.embed.weight.copy_(
                _draw_full((self.spec.vocab, d), seed, "embed"))
            self.pos.copy_(_draw_full((1, self.spec.seq, d), seed, "pos"))
            self.head.weight.copy_(
                _draw_full((self.spec.vocab, d), seed, "head"))
            for i, blk in enumerate(self.blocks):
                _seed_block(blk, self.spec, seed, i)

    def forward(self, idx):
        if self.cp_size > 1:
            s_local = idx.shape[1]
            off = self.cp_rank * s_local
            pos = self.pos[:, off:off + s_local]
        else:
            pos = self.pos[:, : idx.shape[1]]
        x = self.embed(idx) + pos
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
# The interposer simulates timing, not payload, so with FAKEGPU_NCCL_SIMULATE_ONLY
# set (the default gate) an all_reduce here moves no values: the VALUES are wrong
# and the TRAFFIC is right, the same bargain data-parallel mode makes. With the
# transport enabled (FAKEGPU_NCCL_SIMULATE_ONLY=0) the values are really summed
# and the TP loss curve must match the unsharded reference -- that is the gate.

class _ReduceSum(torch.autograd.Function):
    """All-reduce as an autograd node: SUM forward, IDENTITY backward.

    The Row-parallel pattern (Megatron f): the forward sums the rank-partials
    into the whole; the backward is the identity -- dL/d(partial) is dL/d(whole)
    broadcast, NOT a second sum. Summing in backward too hands every rank
    tp x the true output-grad (measured: step-1 loss 8.363 ref vs 8.350 TP).
    The backward collective of a TP sub-layer lives in the Column path
    (_ColLinear), not here. A bare `dist.all_reduce` inside forward is
    invisible to autograd, hence this node. Timing traffic: TP forward ARs
    only from this node; backward ARs from _ColLinear -- one per sub-layer
    per pass, Megatron's schedule, same bytes each.
    """

    @staticmethod
    def forward(ctx, x, group):
        y = x.clone()
        if group is not None:
            dist.all_reduce(y, group=group)
        return y

    @staticmethod
    def backward(ctx, grad):
        return grad, None


class _ColLinear(torch.autograd.Function):
    """Column-parallel linear with a backward all-reduce (gpu-sim-f50).

    Forward needs no communication (input replicated, output sharded). The
    input gradient is partial (sharded grad_out through sharded W) and must
    be SUMMED across the tp group -- without this, TP backward traffic is
    understated by half and, with the transport on, the values are wrong.
    With the transport off the reduction is the usual timing event.
    """

    @staticmethod
    def forward(ctx, x, weight, group):
        ctx.group = group
        ctx.save_for_backward(x, weight)
        return x @ weight.t()

    @staticmethod
    def backward(ctx, grad_out):
        x, weight = ctx.saved_tensors
        gx = grad_out @ weight
        if ctx.group is not None:
            dist.all_reduce(gx, group=ctx.group)
        gw = grad_out.reshape(-1, grad_out.shape[-1]).t() @ \
            x.reshape(-1, x.shape[-1])
        return gx, gw, None


class _Col(nn.Module):
    """Column-parallel: weight [out/tp, in]; output sharded on the last dim."""

    def __init__(self, in_features: int, out_features: int, group=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.normal_(self.weight, std=0.02)
        self.group = group

    def forward(self, x):
        return _ColLinear.apply(x, self.weight, self.group)


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
    def __init__(self, spec: ModelSpec, tp_size: int, group,
                 tp_rank: int = 0, weight_seed: int = 0,
                 layer_idx: int = 0, cp_size: int = 1, cp_rank: int = 0,
                 cp_peers=None):
        super().__init__()
        d, d_ff = spec.d_model, spec.d_ff
        self.n_heads = spec.n_heads // tp_size
        self.head_dim = d // spec.n_heads
        self.qkv = _Col(d, 3 * d // tp_size, group)
        self.proj = _Row(d // tp_size, d, group)
        self.ff1 = _Col(d, d_ff // tp_size, group)
        self.ff2 = _Row(d_ff // tp_size, d, group)
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.cp_size = cp_size
        self.cp_rank = cp_rank
        self.cp_peers = cp_peers or [0]
        # Deterministic sharding (gpu-sim-f50): every rank draws the FULL
        # weight from the same per-layer generator and slices its shard.
        # Same (weight_seed, layer) -> same full weight on every rank, so
        # the reference gate constructs identical weights by construction.
        # Traffic is unaffected (shapes never changed).
        _seed_block(self, spec, weight_seed, layer_idx, tp_size, tp_rank)

    def forward(self, x):
        b, s, d = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h).view(b, s, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        hd = self.head_dim
        cp = self.cp_size
        if cp <= 1:
            qh = q.transpose(1, 2)
            kh = k.transpose(1, 2)
            vh = v.transpose(1, 2)
            scores = (qh @ kh.transpose(-2, -1)) / (hd ** 0.5)
            mask = torch.arange(s, device=x.device)[None, None, None, :] <= torch.arange(s, device=x.device)[None, None, :, None]
            scores = scores.masked_fill(~mask, float("-inf"))
            attn = torch.softmax(scores, dim=-1) @ vh
            att = attn.transpose(1, 2).reshape(b, s, self.n_heads * hd)
        else:
            import torch.distributed as dist
            seq_full = s * cp
            full_k = torch.zeros(b, seq_full, self.n_heads, hd,
                                 device=x.device, dtype=k.dtype)
            full_v = torch.zeros(b, seq_full, self.n_heads, hd,
                                 device=x.device, dtype=v.dtype)
            s = x.shape[1]
            off = self.cp_rank * s
            full_k[:, off:off+s] = k.detach().clone()
            full_v[:, off:off+s] = v.detach().clone()
            for t in range(cp):
                if t == self.cp_rank:
                    continue
                peer = self.cp_peers[t]
                recv_k = torch.empty_like(k)
                recv_v = torch.empty_like(v)
                sk = dist.isend(k.detach().contiguous(), dst=peer)
                rk = dist.irecv(recv_k, src=peer)
                sv = dist.isend(v.detach().contiguous(), dst=peer)
                rv = dist.irecv(recv_v, src=peer)
                sk.wait(); rk.wait(); sv.wait(); rv.wait()
                off_t = t * s
                full_k[:, off_t:off_t+s] = recv_k
                full_v[:, off_t:off_t+s] = recv_v
            qpos = torch.arange(self.cp_rank * s, (self.cp_rank + 1) * s, device=x.device)
            qh = q.transpose(1, 2)
            kh = full_k.permute(0, 2, 1, 3)
            vh = full_v.permute(0, 2, 1, 3)
            scores = (qh @ kh.transpose(-2, -1)) / (hd ** 0.5)
            mask = torch.arange(seq_full, device=x.device)[None, None, None, :] <= qpos[None, None, :, None]
            scores = scores.masked_fill(~mask, float("-inf"))
            attn = torch.softmax(scores, dim=-1) @ vh
            att = attn.transpose(1, 2).reshape(b, s, self.n_heads * hd)
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

    def __init__(self, spec: ModelSpec, tp_size: int = 1, group=None,
                 tp_rank: int = 0, weight_seed: int = 0,
                 cp_size: int = 1, cp_rank: int = 0, cp_peers=None):
        super().__init__()
        self.spec = spec
        self.tp_size = tp_size
        self.cp_size = cp_size
        self.cp_rank = cp_rank
        self.cp_peers = cp_peers or [0]
        self.embed = nn.Embedding(spec.vocab, spec.d_model)
        self.blocks = nn.ModuleList(BlockTP(spec, tp_size, group, tp_rank,
                                            weight_seed, i,
                                            cp_size=cp_size, cp_rank=cp_rank,
                                            cp_peers=cp_peers)
                                    for i in range(spec.n_layers))
        self.lnf = nn.LayerNorm(spec.d_model)
        self.head = nn.Linear(spec.d_model, spec.vocab, bias=False)
        self.pos = nn.Parameter(torch.zeros(1, spec.seq, spec.d_model))
        # Replicated weights drawn identically on every rank (same names as
        # the reference seed_weights, so they agree by construction).
        with torch.no_grad():
            d = spec.d_model
            self.embed.weight.copy_(
                _draw_full((spec.vocab, d), weight_seed, "embed"))
            self.pos.copy_(_draw_full((1, spec.seq, d), weight_seed, "pos"))
            self.head.weight.copy_(
                _draw_full((spec.vocab, d), weight_seed, "head"))

    def forward(self, idx):
        if self.cp_size > 1:
            s_local = idx.shape[1]
            off = self.cp_rank * s_local
            pos = self.pos[:, off:off + s_local]
        else:
            pos = self.pos[:, : idx.shape[1]]
        x = self.embed(idx) + pos
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


# --------------------------------------------------------------------------
# Pipeline parallelism
# --------------------------------------------------------------------------
#
# A stage owns a contiguous slice of layers: stage s holds
# [s*n_layers/pp, (s+1)*n_layers/pp). Stage 0 also owns the embedding, the
# last stage the final norm and head. Stages exchange activations forward and
# gradients backward with blocking send/recv (the GPipe order: all forwards,
# then all backwards); the schedule itself lives in train.py, this class is
# only the per-stage sub-model.
#
# Same timing-only bargain as TP: send/recv move NO payload (the receiver
# keeps its init value), so stage outputs are garbage and the loss in PP mode
# is not meaningful. Traffic and timing are real; the arithmetic is not.

class PPStage(nn.Module):
    """One pipeline stage of TinyLM, optionally tensor-sharded within."""

    def __init__(self, spec: ModelSpec, stage_idx: int, pp: int,
                 tp_size: int = 1, group=None, tp_rank: int = 0,
                 weight_seed: int = 0):
        super().__init__()
        assert pp > 1, "pp==1 needs no stage model"
        assert spec.n_layers % pp == 0, (
            f"n_layers {spec.n_layers} not divisible by pp {pp}")
        self.spec = spec
        self.stage_idx = stage_idx
        self.pp = pp
        self.tp_size = tp_size
        n_local = spec.n_layers // pp
        lo, hi = stage_idx * n_local, (stage_idx + 1) * n_local
        blk = BlockTP if tp_size > 1 else Block
        if tp_size > 1:
            self.blocks = nn.ModuleList(
                blk(spec, tp_size, group, tp_rank, weight_seed, i)
                for i in range(lo, hi))
        else:
            self.blocks = nn.ModuleList(blk(spec) for _ in range(lo, hi))
        self.is_first = stage_idx == 0
        self.is_last = stage_idx == pp - 1
        if self.is_first:
            self.embed = nn.Embedding(spec.vocab, spec.d_model)
            self.pos = nn.Parameter(torch.zeros(1, spec.seq, spec.d_model))
            nn.init.normal_(self.pos, std=0.02)
        if self.is_last:
            self.lnf = nn.LayerNorm(spec.d_model)
            self.head = nn.Linear(spec.d_model, spec.vocab, bias=False)
        self.seed_weights(weight_seed, tp_rank)

    def seed_weights(self, seed: int, tp_rank: int = 0) -> None:
        """Deterministically fill this stage's params (gpu-sim-f50).

        Global layer indices, so stages agree with the unsharded reference
        (which seeds the same names) by construction. TP slices when asked.
        """
        n_local = self.spec.n_layers // self.pp
        lo = self.stage_idx * n_local
        with torch.no_grad():
            d = self.spec.d_model
            if self.is_first:
                self.embed.weight.copy_(
                    _draw_full((self.spec.vocab, d), seed, "embed"))
                self.pos.copy_(
                    _draw_full((1, self.spec.seq, d), seed, "pos"))
            if self.is_last:
                self.head.weight.copy_(
                    _draw_full((self.spec.vocab, d), seed, "head"))
            for j, blk in enumerate(self.blocks):
                _seed_block(blk, self.spec, seed, lo + j,
                            self.tp_size, tp_rank)

    def embed_input(self, idx):
        """First stage only: tokens -> embeddings."""
        return self.embed(idx) + self.pos[:, : idx.shape[1]]

    def run_blocks(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x

    def head_loss(self, x, target):
        """Last stage only: norm + head + next-token cross entropy."""
        logits = self.head(self.lnf(x))
        return torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), target.reshape(-1))

    def local_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
