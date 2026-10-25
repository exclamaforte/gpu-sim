# gpu-sim-fzd: CP gradient sync reduce-scatter

## Equivalence
Both modes produce identical loss curves:
- allreduce (crude): rank0 5.760 -> 5.743, rank1 5.695 -> 5.680
- reducescatter (equivalence-preserving): rank0 5.760 -> 5.743, rank1 5.695 -> 5.680

## Byte cost
MEASURED bytes per rank are EQUAL for both modes in this harness: the reducescatter
path does reduce_scatter_tensor followed by all_gather to reconstruct the full
gradient, so the total traffic equals the allreduce path. The model's
"production" cp_grad (2*(cp-1)/cp * bytes/cp) is NOT exercised by this harness
because it requires a sharded optimizer (each rank updates only its shard) which
this harness does not implement.

## Choice (committed): (b) relabel honestly
The harness's reducescatter path is an equivalence-proven alternative with
identical byte cost. The model's production cp_grad remains a hypothetical
that requires a sharded optimizer. The phase map quotes the crude cp_grad
(conservative). No feasibility change.

## SDPA backend
flash_sdp=True, mem_efficient_sdp=True, math_sdp=True (all enabled).

## Measured bytes per rank (interposer trace, 5 steps, 2 layers, d=32, vocab=256, seq=64, cp=2)
- allreduce (crude): cp_grad allreduce = 519,940 bytes/rank (64 calls)
- reducescatter (equivalence-preserving): cp_grad reduce_scatter + all_gather = 519,936 bytes/rank (63+63 calls)
- Measured bytes per rank are EQUAL for both modes (reduce_scatter + all_gather = allreduce traffic).
- K/V ring traffic (12 reducescatter + 12 allgather) = 380,928 bytes/rank in both modes.
