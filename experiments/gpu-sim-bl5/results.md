Cross-bead citations: experiments/gpu-sim-pkb/traces/trace_S393216_n2_bl5-m1-1792601705.log experiments/gpu-sim-pkb/traces/trace_S393216_n2_bl5-m2-1792601716.log experiments/gpu-sim-pkb/traces/trace_S393216_n2_bl5-m3-1792601721.log

# gpu-sim-bl5: S=384KiB (393216 bytes) under-send investigation

## Status: cause undetermined; see "What would decide it"

## Runs (all committed under experiments/gpu-sim-pkb/traces/)

Race-fixed start order (rank 1 first), 2 s sleep (bl5-m1/m2/m3):

| run | rank0 isends | rank0 bytes | rank1 isends | rank1 bytes | trace |
|-----|-------------:|------------:|-------------:|------------:|
| bl5-m1 | 5 | 327680 (0.833S) | 4 | 262144 (0.667S) | experiments/gpu-sim-pkb/traces/trace_S393216_n2_bl5-m1-1792601705.log |
| bl5-m2 | 5 | 327680 (0.833S) | 4 | 262144 (0.667S) | experiments/gpu-sim-pkb/traces/trace_S393216_n2_bl5-m2-1792601716.log |
| bl5-m3 | 5 | 327680 (0.833S) | 4 | 262144 (0.667S) | experiments/gpu-sim-pkb/traces/trace_S393216_n2_bl5-m3-1792601721.log |

Long sleep (5 s, bl5-long1/long2): identical (5/4, 0.833/0.667). More sleep did not
capture additional sends.

Reversed start order (rank 0 first, bl5-rev1/rev2/rev3): the asymmetry FLIPS which
rank gets 5 vs 4, but the totals (5+4=9 chunks) are identical. Total per run is
stable at 9 chunks = 589824 bytes = 1.5S, vs 12 chunks = 786432 expected.

NCCL reports success in every run: "rank 0 allreduce sum[0]=3.0 (expect 3.0) OK"
and "rank 1 allreduce sum[0]=3.0 (expect 3.0) OK".

## Findings

1. The under-send is STABLE (9 chunks, not 12) regardless of start order or sleep
   duration. It is not a capture race (the race affects which rank gets which
   count, not the total).
2. NCCL reports success (sum=3.0) in every run, so if the plugin silently
   dropped sends the allreduce result would be wrong -- it is not. This argues
   against a silent under-send in the plugin and toward incomplete capture or
   NCCL's own chunking.
3. 384KiB = 6 x 65536 exactly. NCCL sends 9 chunks of 65536 = 589824 bytes
   total, 196608 bytes (3 chunks) short of the expected 786432.

## Named cause: undetermined

Candidate: incomplete capture. The plugin traces isends on the single link
(link=0). NCCL may send some data via a path the plugin does not trace (in-place,
same-rank, or a second channel not recorded), or may use a chunk count that
does not correspond to isend count. The isend count alone cannot distinguish
these.

## What would decide it

A capture that records ALL fabric transfers (both ranks, both directions, every
channel) and compares total traced bytes to the expected 2S=786432. Alternatively,
an NCCL-side byte counter. If the complete capture still shows 9 chunks, the
cause is NCCL's chunking for this size; if it shows 12, the current capture is
incomplete and prior factor measurements that depend on isend counts need audit.

## Escalation

No silent under-send is evident (NCCL reports success). No separate fix bead
filed. 384KiB stays excluded from canonical tables and model evidence.

## Regenerability

bl5-m1/m2/m3 all agree (5/4 chunks). Reversed runs agree on totals (9 chunks).
Long sleep did not change counts.
EOF
echo written
## ESCALATION UPDATE (decisive value test)

The byte-count analysis above is superseded. A value-correctness test (fill rank0=1.0, rank1=2.0, expect 3.0 on all elements) gives:

- S=393216 (384KiB): ncclAllReduce returns ncclSuccess but the operation NEVER
  completes (cudaMemcpy blocks forever). Both ranks hang. This is "success reported
  for incomplete work" -- the escalation case.
- S=524288 (512KiB, control): VALUE OK on both ranks (all elements == 3.0).

Outcome (a): real size-dependent simulator defect. The 9-chunk trace was partial
progress before the hang, not a completed (if short) allreduce. 384KiB stays
excluded from canonical tables. Filed as P1 bead gpu-sim-7vw.

The earlier "no silent under-send" conclusion was wrong: a chunk count cannot see
whether data moved. The decisive test is value correctness.
