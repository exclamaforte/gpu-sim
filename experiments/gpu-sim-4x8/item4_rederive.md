# gpu-sim-4x8 item 4: re-derive the FSDP micro-benchmark row with the predictor fix

The bead closed with "FSDP comm_sim_ms re-derive deferred (requires GPU run)". The GPU was
idle, so this is that run. Interposer built from 7114575 (`make`), topology emitted by
`tools.torch_parallel.topology.emit(4, ..., 'nvlink')`.

## Direct trace (world=4, tp=1 pp=1 dp=4, zero=3, 3 steps, rank 0, grank==0)

    allgather       bytes 21514240   predicted_s 1.8928533e-05
    reducescatter   bytes 21514240   predicted_s 1.8928533e-05

* `bytes` is now the **total** payload (21.5 MB), not the per-rank shard (5.4 MB): the trace
  change in 043d301 is real.
* `predicted_s` = 18.9 us for 21.5 MB total at n=4 -> per-rank ring traffic (3/4)*21.5 =
  16.1 MB in 18.9 us = **853 GB/s**, i.e. the total is what the cost model was given.
  Pre-fix the same op was priced from the shard (5.4 MB -> ~4.7 us), so the fix is a genuine
  4x change at dp=4 -- verified directly rather than inferred from an aggregate.

## Aggregate (the row docs section 5 quotes)

    python3 -m tools.torch_parallel.sweep --world-sizes 4 --zero 3 --steps 5 --fabric nvlink
    world  tp pp dp zero  step_ms  coll_msgs  coll_MB  p2p_msgs  p2p_MB  comm_sim_ms  peak_MB
        4   1  1  4    3     32.2          5     86.1         0      0.0         0.08    785.6

`coll_MB` moved 21.5 -> 86.1 (the 4x total-vs-shard change); `comm_sim_ms` 0.09 -> 0.08 is
consistent with 5 collectives x 18.9 us = 0.095 ms and does not, on its own, show the fix --
which is why the direct trace above is the evidence.
