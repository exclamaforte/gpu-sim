# gpu-sim-u2o.9.4 — the analytic roofline baseline: the number the learned model must beat

This bead produces a REPORT, not a model change. Nothing under `epic_u2o/`
was modified: the baseline stands as 46p left it. Every checkable figure
below is verified by `report_verify.py` against COMMITTED paths
(`metrics.json` in this directory records the reconciliation).

## The number, per regime (46p's table, carried not fitted)

| regime (bottleneck transition) | n | mean abs err | signed bias |
|---|---|---|---|
| latency->latency (latency-bound) | 220 | 0.5638 | **-0.5638, all 220 underpredict** |
| latency->memory (memory via the latency map) | 204 | 0.6873 | +0.3780 |
| memory->memory (memory-bound throughout) | 32 | 1.1209 | **+1.1164, 29 of 32 overpredict (3 underpredict)** |
| **overall** | 456 | 0.6581 | -0.0245 |

Overall arc: **7.7238 -> 0.6581 (11.7x)**, by removing four wrong
assumptions (register formula, invented latency terms, dead read/write
fields, double-discounted bandwidth) and fitting nothing. The step
0.7258 -> 0.9955 between zhq and 46p is the 8.3 memory arm arriving WITH
the occupancy derate still on; 46p removed the derate.

## The cancellation warning (acceptance item 8)

**The overall bias of -0.0245 is cancellation, not accuracy, and reporting
it as accuracy would be the single most misleading thing this report could
do.** The latency regime underpredicts by 0.56 while the two memory
regimes overpredict by 0.38 and 1.12. A learned model clearing "overall
bias near zero" has cleared nothing: 9.1 must be judged per regime.

## Residual A: memory->memory overpredicts by 1.12

Primary candidate, with evidence: **cache-resident traffic**. Effective
rates (bytes/median, cache-inclusive) exceed DRAM read-peak on 148/456
rows (max 7480 GB/s), all at large n across all ops: 119/204 in
latency->memory, 29/32 in memory->memory, 0/220 in latency->latency.
Repeated trials on resident buffers serve issued traffic from L2, so a
candidate is DEMOTED on sign (slower writes predict underprediction; 29 of
32 rows overpredict, 3 underpredict -- the demotion holds but is weaker than
an "all" would warrant). Scalar overcount (alpha as DRAM) contributes at most
~25% for saxpy-class rows. Follow-up: a cache-busting streaming
measurement by arity. The memory arm's form does not change in this bead.

## Residual B: latency->latency underpredicts by 0.56

Candidates: the n-local floor (39u ruled it out overall, not for this
subset where the floor dominates), or a fixed per-launch cost above the
floor (39u mechanism (a), named, never tested). Carry, do not fit.

## The contention bound (acceptance item 5): resolved on the corpus side

456/456 corpus cells carry `queue_state` with `n_other_running == 0` AND
`n_other_active == 0` (both corpora, checked live from the committed
rows). No cell used by this baseline ran contended. The ceiling anchor
(megakern `rtx5090.json`, ts `20261003t213327-db6b7f3f`) records NO
contention state: stated, not assumed. Item 7 honored throughout: the
bound uses queue_state, never clocks.

## The ceiling, as an argmax (acceptance item 2)

Candidate set (membership: measured on this silicon or vendor theory from
the same file; nothing else admitted):

| value | source | standing |
|---|---|---|
| 1688.592 GB/s | megakern peak_read | CHOSEN (46p form a): achieved at saturation |
| 1792.0 GB/s | megakern theoretical | REJECTED: theory, no measured efficiency |
| 1468.996 GB/s | megakern peak_copy | FOLLOW-UP (residual A): mixed-traffic ceiling by arity, to be measured |

Argmax rule: a ceiling is a rate some launch demonstrably sustains, not a
rate none has reached.

## Conventions and provenance (acceptance items 1, 4, 6, 8)

- Bytes convention, beside every figure: 8.3 arity counts per row
  (subscripts + `*p` derefs + atomics + float scalars, x4 bytes, x covers;
  loop/branch-resident counted once). Exemplar, recomputed from committed
  bytes and committed median: scale n=2^22 row `66a4b75ee7ee19a8`,
  bytes 50331648, median 0.033536 ms, achieved 1500.8 GB/s.
- Per-regime median effective rates (same convention): latency->latency
  46.9 GB/s (max 538.7, 0 above peak); latency->memory 1977.5 (max 7479.8,
  119 above peak); memory->memory 3624.1 (max 6050.2, 29 above peak).
- Families: 8 ops x 57 rows, ONE kernel per op. Per-op figures are
  per-kernel measurements wearing family labels; 9.1 judges across 8
  kernels, not 8 families.
- Regime weighting is explicit: the three rows above ARE the weighting; no
  single global bar is published. Discard thresholds: none beyond the
  corpus's own construction -- excluded rows: 0 (all 456 medians > 0, all
  used). Anchor: the 456-row corpus (7.1 ts `20260928t074004-b429f469` +
  e5q ts `20261010t224024-465ad815`), tree at the 46p merge; model code
  untouched by this bead.

## Recompute

`bash run.sh`: arc reconciliation + contention census + families +
regime/achieved recomputation from committed paths, byte-compares
`metrics.json`. CPU-only.
