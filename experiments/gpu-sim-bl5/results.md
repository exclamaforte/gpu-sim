Cross-bead citations: experiments/gpu-sim-pkb/traces/trace_S393216_n2_bl5-m1-1792601705.log experiments/gpu-sim-pkb/traces/trace_S393216_n2_bl5-m2-1792601716.log experiments/gpu-sim-pkb/traces/trace_S393216_n2_bl5-m3-1792601721.log experiments/gpu-sim-bl5/metrics.json

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

NOTE: an earlier draft quoted "rank 0 allreduce sum[0]=3.0 (expect 3.0) OK".
That line is produced by experiments/gpu-sim-m3d.8/allreduce_test.c (commit 1846328),
which hardcodes n=1024 (4096 bytes) and verifies every element. It is a 4KiB result,
not 384KiB, and is not evidence about this size. Cited only as proof that the sim
fabric reduces correctly at small sizes.

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
## Value-correctness experiment (decisive)

Harness: experiments/gpu-sim-m3d.8/allreduce_test_size.c (size from argv[4]), with
the element check ported from allreduce_test.c: fill the whole buffer (rank0=1.0,
rank1=2.0), run ncclAllReduce(ncclSum), cudaDeviceSynchronize, then verify every
element == 3.0. Built and run via run_measure.sh (sim fabric, topo_single_eut.bin).
Prints "about to cudaDeviceSynchronize" before the sync and "copy-back done" after.

- S=393216 (384KiB): HANG. Both ranks print "about to cudaDeviceSynchronize" and
  then stop (no "synchronize done", no "copy-back done", no VALUE line; exit 124).
  The allreduce never completes. Byte trace shows 5 sends rank0 / 4 sends rank1.
  Committed evidence: traces/trace_S393216_n2_val384f.log,
  traces/run_val384f_rank0.txt, traces/run_val384f_rank1.txt.
- S=524288 (512KiB, control): NONDETERMINISTIC. val512d (commit 2b88120): MISMATCH
  (16384/131072 wrong, first idx=57344 got=5.0). val512f (commit b384116): VALUE OK
  (all 131072 elements == 3.0). Same fabric code; only stderr fprintf timing differed.
  Uniform fill is structurally blind to misdelivery (identical payloads make swaps
  invisible). Committed evidence: traces/trace_S524288_n2_val512d.log (MISMATCH),
  traces/trace_S524288_n2_val512f.log (OK), traces/run_val512d_rank*.txt,
  traces/run_val512f_rank*.txt.

Conclusion: at 384KiB the sim fabric allreduce hangs (operation never completes,
success is reported by the async ncclAllReduce return). At 512KiB it completes
but produces NONDETERMINISTIC values (val512d MISMATCH, val512f OK). The uniform
fill used in val512f is structurally blind to misdelivery (identical payloads make
swaps invisible); the non-uniform fill in val512d exposes it. The defect is NOT
size-dependent in the way previously concluded: 384KiB always stalls, 512KiB
sometimes produces wrong values. The "512KiB passes" conclusion is WITHDRAWN.
Filed as P1 bead gpu-sim-7vw. 384KiB stays excluded from canonical tables/model
evidence. 512KiB value-correctness is nondeterministic and must not be cited as
passing until the non-uniform fill test (gpu-sim-7vw) is clean.

Note: the earlier "NCCL reports success: sum[0]=3.0 OK" quote in this document was
from experiments/gpu-sim-m3d.8/allreduce_test.c (commit 1846328), which hardcodes
n=1024 (4096 bytes). It is a 4KiB result, not 384KiB, and is not evidence about
this size.
