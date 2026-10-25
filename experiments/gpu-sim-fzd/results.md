# gpu-sim-fzd: CP gradient sync reduce-scatter

## Equivalence
Both modes produce identical loss curves:
- allreduce (crude): rank0 5.760 -> 5.743, rank1 5.695 -> 5.680
- reducescatter (production): rank0 5.760 -> 5.743, rank1 5.695 -> 5.680

## Byte counts (model)
- Crude (allreduce): 2*(cp-1)/cp * P/(tp*pp)*dtype bytes per rank
- Production (reducescatter): 2*(cp-1)/cp * P/(tp*pp)*dtype/cp bytes per rank

## Modelling consequence
The model's cp_grad_mode flag selects crude (default) or production. The harness
now defaults to reducescatter (production). Under production, cp=4 comes back to
~81 ms vs FSDP 73 ms -- still behind, but the gap is K/V traffic, not sync.
The phase map quotes crude (conservative). Both numbers are reported; the
recommendation (FSDP first) does not change.
