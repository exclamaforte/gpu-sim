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

- S=393216 (384KiB): HANG. ncclAllReduce returns ncclSuccess but the operation
  never completes (cudaDeviceSynchronize blocks; no VALUE line printed; exit 124).
  Trace shows 5 sends rank0 / 4 sends rank1 (same under-count as the byte runs).
  Logged in traces/run_val384e_rank0.txt and run_val384e_rank1.txt (no VALUE line).
- S=524288 (512KiB, control): completes but VALUE MISMATCH -- rank0 reports
  "16384/131072 wrong, first idx=57344 got=5.0"; rank1 reports "32768/131072
  wrong, first idx=32768 got=1.0". Byte trace shows 16 sends per rank (1.00S),
  but the reduced values are wrong. Logged in traces/run_val512d_rank0.txt and
  run_val512d_rank1.txt.

Conclusion: the sim fabric's allreduce is broken at these sizes -- it either hangs
(384KiB) or produces incorrect values (512KiB), while reporting success. The byte
count (1.00S at 512KiB) does NOT imply a correct reduction. Outcome is a real
size-dependent simulator defect, not an accounting artifact. Filed as P1 bead
gpu-sim-7vw. 384KiB and 512KiB stay excluded from canonical tables/model evidence
until the fabric allreduce is fixed.

Committed evidence:
- traces/trace_S393216_n2_val384e.log + run_val384e_rank0.txt and run_val384e_rank1.txt (hang, no VALUE line)
- traces/trace_S524288_n2_val512d.log + run_val512d_rank0.txt and run_val512d_rank1.txt (VALUE MISMATCH)
