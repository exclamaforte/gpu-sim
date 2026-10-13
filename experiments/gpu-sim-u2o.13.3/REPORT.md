# gpu-sim-u2o.13.3 REPORT: search machinery shipped, validation gated

CPU-only bead, no GPU.

## Shipped (real, tested)

epic_u2o/surrogate_search.py: rank_configs (greedy/UCB) + regret().
Perfect surrogate recovers the optimum with regret 0; noisy stays
bounded; UCB demonstrably explores the uncertain arm greedy skips.

## Gate (test-pinned, the finding)

Scanned every measured file (5): max 1 config per kernel corpus-wide.
Within-kernel tuning sets do not exist -- the "search launch configs"
half has no real input today. Opening needs a multi-config measurement
campaign on the real GPU plus a cost model that beats GBT (11.4: neither
holds). The search code waits ready, not rotting: its contract is tested.

## Backing doc

deferred.md: only the 13.3 placeholder replaced.
