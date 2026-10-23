# gpu-sim-tbp results

Convention (stated, enforced): true DDP -- disjoint rank shards of one seeded
batch + explicit 1/dp averaging (--dp-mean); transport-on without it refuses
loudly. Same-init on all ranks under --dp-mean; dropout is 0.0.

- EQUIV (4df16d5d): world1 vs world2-SGD-lr0.01-seed7, 12 steps.
  world2-mean matches world1 to <=1.13e-07 every step (per-rank losses
  legitimately differ: local-shard means).
- LONG (b51a1e24): world2 50 steps 8.4806 -> 8.4276, 0 up-steps, wall 9s.
- ENFORCE (4f5f8c4b): no---dp-mean + transport -> SystemExit refusal, rc=1.
- Transport race fix (link-publish) found here at k=26 (both ranks
  arrived=1): publish under pid-unique temp + link(); join path only sees
  initialized headers. n5h review note filed on gpu-sim-n5h.
Evidence: tbp-w1.json tbp-w2.json tbp-w2long.json; checks: tbp_check.py.
