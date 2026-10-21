# gpu-sim-eut pre-registration — written before any measurement

## 1. Claim under test

Re-measure the canonical byte table (n=2, six sizes) with committed raw traces
and an explicit reduction rule, because the prior table
(experiments/gpu-sim-pkb/canonical_byte_table.md) agrees with its own cited
traces on NO row (PI review commit 7bb3764).

## 2. Sizes (fixed)

S (bytes) in {4096, 65536, 262144, 524288, 1048576, 2097152}
(floats = S/4, dtype float32).

## 3. Reduction rule (phase boundary) — asserted in code, not prose

A phase boundary is a change in the isend digest. Concretely:
- Each trace line "isend rank=R ... bytes=B ... digest=D" records one isend.
- Group consecutive isends by rank; a new phase begins when digest D differs
  from the previous isend's digest for that rank.
- per-phase pieces = number of isends in that phase (for that rank).
- per-phase bytes = sum of B over isends in that phase.
- total bytes/rank = sum of B over all isends for that rank.
- byte factor = total bytes/rank / S.
- number of phases = number of distinct digests for that rank.

This rule is implemented in reduce_trace.py::reduce() and asserted there.

## 4. Completeness assertion per trace (INVALID if any fails)

For each rank in the trace:
- total bytes/rank >= S  (a real allreduce cannot send less than the buffer)
- total bytes/rank <= 4*S (n=2 ring allreduce sends at most 2 phases x 2 directions)
- distinct digests == 2 (the two phases: reduce-scatter + allgather) for a
  successful n=2 ring allreduce.
A trace failing any of these is reported INVALID, not reduced.

## 5. Expected per size (prior, UNVERIFIED — hypothesis, not assertion)

Prior table claimed byte factor 2.0 for S <= 262144 and 1.0 for S >= 524288.
Re-measurement will confirm or refute; if it refutes, that is the finding and
gpu-sim-ggh closes as NOT-A-DEFECT.

## 6. Regenerability

A fresh re-run of one size must reproduce committed numbers within 0% (byte
counts are deterministic given the same NCCL heuristic) OR the difference is
explained. Counts (pieces) are deterministic; durations are not compared.

## 7. Verdict rule

- STEP-CONFIRMED if the re-measured byte factor shows a 2.0 -> 1.0 step
  between 256 KiB and 512 KiB (with committed trace per size).
- NO-STEP otherwise; then gpu-sim-ggh closes as NOT-A-DEFECT citing this bead.
EOF
echo written
## 8. Evidence-integrity rules (added per PI review)

- No committed file may be modified by this bead. All measurement outputs are new
  paths under experiments/gpu-sim-pkb/traces/ with the naming pattern
  trace_S<S>_n<nranks>_<runid>.log, each carrying its own S, nranks and run id
  in a self-describing header. Per-run stdout/stderr go to
  traces/run_<RUN_ID>_rank{0,1}.txt. Committed paths (run0.log, run1.log) are
  never written.
- Measuring harness: experiments/gpu-sim-m3d.8/allreduce_test_size.c (new code
  in this bead). It takes per-rank element count as argv[4], allocates
  n*sizeof(float) per rank, calls ncclAllReduce, and writes only to the trace
  path given by GPUSIM_TRACE. It does not modify the model or timing code.
- Completeness gate: total bytes per rank must be an exact multiple of S with
  factor in [1,4]. Non-integer factors (e.g. 1.25) are INVALID, not rounded.
