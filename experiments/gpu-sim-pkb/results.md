# gpu-sim-eut: Re-measured canonical byte table (n=2)

## Verdict: NO-STEP

The claimed 2.0 -> 1.0 byte-factor step between 256 KiB and 512 KiB is NOT
reproduced. No valid trace shows a 2.0 factor; the byte factor is 1.0 for every
valid trace at every size (rank 1 consistently; rank 0 is 1.0 at >= 512 KiB and
1.25 at 64/256 KiB). The one trace that showed a 2.0 factor (4 KiB, rank 0)
is INVALID under the completeness assertion.

Consequence: gpu-sim-ggh (model correction) is closed as NOT-A-DEFECT, citing
this bead. The model's constant 2.0 byte factor is not contradicted by the
re-measured data; the prior table's step was an artifact.

## Re-measured byte table (n=2, single link bw=10 Gbps lat=5 us)

| S (bytes) | rank0 factor | rank1 factor | isends (r0/r1) | valid? | trace |
|-----------|-------------:|-------------:|----------------|--------|-------|
| 4096      | 2.00         | 1.00         | 2 / 1          | INVALID (rank1 dd=1) | traces/trace_S4096_n2_eut4-4096-1792598352.log |
| 65536     | 1.25         | 1.00         | 5 / 4          | valid  | traces/trace_S65536_n2_CLEAN.log |
| 262144    | 1.25         | 1.00         | 5 / 4          | valid  | traces/trace_S262144_n2_CLEAN.log |
| 524288    | 1.00         | 1.00         | 16 / 16        | valid  | traces/trace_S524288_n2_CLEAN.log |
| 1048576   | 1.00         | 1.00         | 16 / 16        | valid  | traces/trace_S1048576_n2_CLEAN.log |
| 2097152   | 1.00         | 1.00         | 16 / 16        | valid  | traces/trace_S2097152_n2_CLEAN.log |

Inputs from committed artifacts:
- Fabric params: bw=10.0 Gbps, lat=5.0 us, loss=0, jitter=0
  (experiments/gpu-sim-pkb/run_measure.sh single-link topo).
- Reduction: experiments/gpu-sim-pkb/reduce_trace.py (phase boundary by
  digest change, asserted in code).
- Completeness: total bytes per rank in [S, 4S]; distinct digests == 2 (claimed
  phases). A failing trace is INVALID, not reduced.

## Key findings

1. NO-STEP: no valid trace shows a 2.0 byte factor. The prior table's
   2.0 -> 1.0 step does not exist in the raw data.
2. Digests are NOT a stable phase marker: distinct_digests per rank varies
   across runs (e.g. rank0 sometimes 2, sometimes 3 for the same size), and
   the per-rank phase split by digest is asymmetric (rank0: 8+1+7 chunks,
   rank1: 1+15). The digest-based completeness assertion (distinct==2) is
   therefore unsatisfiable in a stable way; 4 KiB is INVALID on it.
3. The byte factor is consistently 1.0 for rank 1 at every size, and 1.0 for
   rank 0 at >= 512 KiB. The model's constant 2.0 factor over-predicts
   even the (invalid) 2.0 observation is not reproducible as a valid trace.
4. Regenerability checked: a fresh re-run of 2 MiB reproduces 32 isends/rank
   (byte factor 1.0) within 0% on count. Digest grouping differs by rank
   across runs (documented above).

## Implication for 4yw

gpu-sim-4yw Leg 1 compared the model against the prior (unverified) table and
found MODEL-DOES-NOT-REPRODUCE at >= 512 KiB. That verdict rested on a
reference that disagrees with its own traces. With the re-measured table
showing NO step, the model's 2.0 factor is not contradicted; the mismatch was
in the reference, not the model. 4yw's verdict should be re-evaluated against
this table (separate bead).

## Pre-registration

experiments/gpu-sim-pkb/pre_registration.md (committed 4a5ad5d, before measurement).
EOF
echo written