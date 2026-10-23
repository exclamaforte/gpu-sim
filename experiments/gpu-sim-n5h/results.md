# gpu-sim-n5h results

All three validations green on GPU (job IDs in bd comments).

- Gate (58b6f1ae): default-unset world-4 replay -> coll=149 MB=86.1,
  byte-exact vs committed experiments/gpu-sim-8y7.3/sweep-zero1.json
  (see gate-world4.txt). step_ms 8.2 vs 14.6 is timing wobble, not gated.
- Exact sum (80108281): ar_probe f32 EXACT + bf16 tol on both ranks,
  asserted against a host-side CPU reference.
- Peer death (af2f7c76): `barrier timeout: op=allreduce k=0 rank=0
  nranks=2 arrived=1 done=0` + SIGABRT rc=134, no hang.
- Negative controls: without transport (d7804aa5) f32 MISMATCH
  err=1024.0 rc=1 (assert guards the contract); pre-stale-fix run
  (e390e1d4) showed arrived=3, fixed by O_EXCL-create +
  join-only-if-creator-alive.
