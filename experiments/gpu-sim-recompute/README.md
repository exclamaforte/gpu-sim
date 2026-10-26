# What activation checkpointing actually costs (calibration for gpu-sim-hz3u)

Measured, not cited: `measure_checkpoint_overhead.py` on the lab's single RTX 5090, bf16, 8-block stacks,
per-block `torch.utils.checkpoint(use_reentrant=False)` (the standard pattern), SGD lr=0, 6 timed iters
after 2 warmup.

| shape | step time, no recompute | with recompute | time ratio | peak mem, no rec | with rec | mem ratio |
|---|---|---|---|---|---|---|
| d=1024, h=8, seq=8192, L=8 | 72.09 ms | 91.47 ms | **1.269** | 2.554 GB | 0.848 GB | **0.332** |
| d=2048, h=16, seq=4096, L=8 | 74.81 ms | 93.21 ms | **1.246** | 3.208 GB | 2.031 GB | **0.633** |
| d=1024, h=8, seq=4096, L=8 | 27.57 ms | 34.23 ms | **1.242** | 1.487 GB | 0.722 GB | **0.486** |

## What the model should take from this

* **Compute**: full per-layer recomputation costs **1.24x-1.27x** step time (24-27% more). The rule of thumb
  ("a third more forward FLOPs") is close but slightly pessimistic; charge the measured range.
* **Memory**: peak memory falls to **33%-63%** of the unrecomputed case, and the saving grows with sequence
  length (33% retained at seq 8192, 49% at 4096, 63% at 4096 with a wider model). So a model term of the
  form `retained = 1 - alpha*recompute` should use **alpha ~= 0.5-0.67**, and should not be a constant
  across seq if that is cheap to express.
* Peak includes weights and gradients, so at very small batches the ratio is bounded below by them: this is
  why the wide model (d=2048) shows the smallest saving.

## Scope, stated so the number is not over-claimed

One GPU, bf16, no TP/PP/CP, one PyTorch version, and `nn.MultiheadAttention` rather than a fused attention
kernel. A production stack with flash-attention recomputes less attention memory and shows a smaller
memory saving than the seq=8192 row here. The TIME ratio is the more transferable of the two.

## Where recomputation actually changes an answer (proxy probe)

`proxy_probe.py` scales `act_factor` by the measured retained fraction (0.33) -- faithful for the MEMORY
term, but it does not charge the 1.25x time, so recompute rows are optimistic in step time.

Result: across 70B/405B x {80, 800} GPUs x batch {8, 32}, recomputation changes NOTHING for 405B (same
config, same margin, 0.2-0.8 GB less memory) because 405B is optimizer-state bound at pp=100/dp=1
(14 B/param x 4.05e9 = 56.7 GB/rank). It changes exactly one answer in the whole grid:

  70B, 80 GPUs, batch=32, no recompute : (tp=1 pp=4 cp=1 dp=20 z3) 2.769 s  58.0 GB  margin 0.45%
  70B, 80 GPUs, batch=32, recompute   : (tp=1 pp=1 cp=1 dp=80 z3) 2.474 s  70.9 GB  margin 10.6%

With recomputation the model stops buying pipeline stages purely to hold activations and buys DATA
parallelism instead: no PP at all, FSDP across all 80 GPUs, 10.6% faster and far better separated (0.45% ->
10.6%). That is the regime real frontier training lives in -- large per-rank batch on a small model -- and
it is the one where the knob earns its place.

I got this wrong twice before measuring: first I said recompute would make 405B feasible at 80 GPUs
(it is optimizer-bound, 70.9 GB floor), then that it would relax pp=100 at 800 (also optimizer-bound). The
proxy grid is what settled it.

## The model's 14 B/param optimizer assumption is probably too high (measure_opt_state_bytes.py)

The 405B feasibility boundary at 80 GPUs rests entirely on the model charging **14 B/param** of optimizer
state (fp32 master 4 + fp32 Adam m/v 8 + bf16 grads 2). Measured on this GPU with a real AdamW step:

  params 5e8, bf16 storage, AdamW defaults, one fwd/bwd/step
  peak allocated 5.00 GB  ->  **10.0 B/param**   (model assumes 14.0)

PyTorch's bf16 AdamW keeps the bf16 parameter AS the master copy; there is no separate fp32 master tensor
unless an AMP-style optimizer adds one. So the 4 B/param the model charges for master weights may not exist
in a real bf16 training loop.

Implication, which is uncomfortable because it would undo part of what §0 currently says: at 10 B/param the
405B state floor at 80 GPUs is 50.6 GB instead of 70.9 GB, and with ~19 GB of activations at pp=10 that is
~70 GB against an 80 GB card -- i.e. 405B@80 may be FEASIBLE, and "infeasible" is an artifact of the model's
optimizer accounting rather than a property of the hardware.

Caveats, so this is not over-read: one tensor, no sharding, defaults only, and peak allocation is not the same
as resident footprint (fragmentation, allocator caching). The first version of this script measured
22.0 B/param because its forward ran in fp32 and materialised a full fp32 copy of the parameter -- a real
bf16 model does not pay that. Both numbers are kept in the commit history for that reason.

## CORRECTION: peak vs steady state, and where the model's 14 B/param actually sits (measure_opt_styles.py)

My earlier note claimed the model's 14 B/param overestimates. Measuring all three real styles, separating
steady state (what training RESIDES at) from peak (what OOMs), says something more useful:

| style | steady B/param | peak B/param | what is resident |
|---|---|---|---|
| bf16 AdamW, bf16 params, fp32 NOT forced | **6.0** | 10.0 | p 2 + bf16 moments 2+2 -- PyTorch keeps Adam states in the PARAMETER dtype |
| AMP classic (fp32 params, autocast bf16, GradScaler) | **12.0** | 20.0 | p 4 + fp32 moments 4+4 |
| bf16 params + explicit fp32 master (DeepSpeed/FSDP style) | **12.0** | 24.0 | p 2 + master 4 + fp32 moments 4+4 = 14 arithmetic, 12 measured steady |

So the model's 14 B/param is:
* slightly ABOVE the fp32-moment steady state (12) -- conservative by 2 B/param, i.e. by ~17%;
* BELOW the measured peak for every fp32-moment style (20-24), because AdamW's fused kernels allocate
  parameter-sized temporaries. Peak is what actually OOMs.

Consequence for the 405B question, with GQA-aware params (405.8e9):
* bf16 moments (6 B steady, 10 B peak): 30 GB steady, 51 GB peak -> FITS at 80 GPUs;
* fp32 moments (12 steady, 20 peak): 61 GB steady, 101 GB peak -> does NOT fit at 80 GPUs.

That is the honest resolution of the "is 405B feasible at 80 GPUs" question: it depends on the optimizer
style, and the model's single 14 B/param number lands between two real regimes whose answers differ.

Two measurement bugs are kept in the history because they were both mine and both instructive: an fp32
forward that materialised an fp32 copy of the parameter (22.0 B/param), and a duplicate optimizer in the
bf16_master row that counted two sets of moments (24.0 peak, though the steady figure survived).

## The peak is partly a CHOICE: fused AdamW cuts it 20 -> 16 B/param (measure_adamw_variants.py)

At fixed fp32 params and fp32 moments (steady 12.0 B/param in every variant):

| AdamW variant | peak B/param | peak GB (5e8 params) |
|---|---|---|
| default (`foreach=True`) | 20.01 | 10.00 |
| `foreach=False` | **24.01** | 12.00 |
| `fused=True` | **16.01** | 8.00 |
| `fused=True, foreach=False` | 16.01 | 8.00 |

So the gap between steady state (12) and peak (20-24) is not physics -- it is the optimizer implementation.
Fused AdamW removes half of it (8 -> 4 B/param), and the plain Python-loop variant is the worst of the three.

Consequence for the boundary this project cares about, 405B x 80 GPUs (5.07e9 params per GPU with GQA-aware
`total_params`):
* steady only:            12 B ->  60.9 GB   fits
* fused AdamW peak:       16 B ->  81.2 GB   infeasible by 1.5%
* default AdamW peak:     20 B -> 101.4 GB   clearly infeasible
* foreach=False peak:     24 B -> 121.7 GB   worst

Which means "does 405B fit on 80 GPUs" is a question about the optimizer implementation AND the moment
precision, not about the hardware alone -- and the honest model answer names both.
