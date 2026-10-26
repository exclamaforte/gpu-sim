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
