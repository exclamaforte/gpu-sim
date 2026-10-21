# PI check of the reference table behind 4yw's Leg 1 — REFERENCE-UNVERIFIED

Written by the PI while reviewing `gpu-sim-4yw`. It does **not** change `4yw`'s code or
its Leg 2; it challenges the ground truth Leg 1 compares against.

## What was checked

`experiments/gpu-sim-pkb/canonical_byte_table.md` is the reference Leg 1 uses. It cites
its raw evidence as:

> Raw traces: /tmp/scratchpad/trace_*.log (regenerable).

Those traces are **outside the repository**, in `/tmp` (a 16 GB memory-backed tmpfs that
this repo has already lost data to). They still existed at review time and are preserved
here in `stale_traces_from_tmp/` before the tmpfs is cleared. `trace_4096.log` was
already absent.

## What the traces say, per rank, nranks=2

| S (bytes) | table: isends/rank | trace: sends/rank | table: total bytes/rank | trace: total bytes/rank | trace: bytes/send |
|---|---|---|---|---|---|
| 65536 | 8 | **8** | 131072 (2S) | **524288 (8S)** | 65536 |
| 262144 | 8 | **16** | 524288 (2S) | **1048576 (4S)** | 65536 |
| 524288 | 16 | **16** | 524288 (S) | **2097152 (4S)** | 131072 |
| 1048576 | 16 | **2** | 1048576 (S) | **8192 (0.008S)** | 4096 |
| 2097152 | 16 | **2** | 2097152 (S) | **8192 (0.004S)** | 4096 |

**The table and its own cited traces agree on no row except the send count at
S = 65536 and S = 524288.** At 1 MiB and 2 MiB the traces contain two 4096-byte sends
per rank — 0.8 % and 0.4 % of the buffer — which is not a complete allreduce at all;
those two files are also timestamped nine minutes after the others (00:40/00:41 vs
00:31), consistent with a later, different run overwriting them.

## The specific claim Leg 1 rests on does not appear in the raw data

Leg 1's verdict is that the measured byte factor **steps from 2.0 to 1.0 at ≥ 512 KiB**,
and that the model's constant 2.0 therefore over-predicts by exactly 2× on three rows.

In the traces the *shape* is identical at 256 KiB and 512 KiB: 16 sends per rank, two
digest groups of 8, **bytes per send = S/4 in both cases**, total **4S in both cases**.
So between 256 KiB and 512 KiB the raw data shows **no step at all** — the factor is
constant. The step is introduced by the table's reduction (its `phases` /`per-phase
pieces` / `distinct bytes` columns), and I could not reproduce that reduction from the
traces by any rule I could justify.

The table's own note already contains the warning: *"the 8sy derivation is NOT
universal; it holds at 4 KiB only"*, and its header argues with itself about whether the
per-rank total is 1× or 2× per direction.

## Consequence for 4yw

**The correct Leg 1 verdict is `REFERENCE-UNVERIFIED`, not `MODEL-DOES-NOT-REPRODUCE`.**

- The comparison *method* is sound and the pre-registration was fixed in advance: this is
  not a complaint about how Leg 1 was run.
- The *reference* is unusable: a table that disagrees with its cited traces on every row,
  whose evidence lives in `/tmp`, and whose own notes record doubt, cannot carry a
  verdict that the model is wrong on three rows — least of all when the one structure the
  traces *do* decide (256 KiB vs 512 KiB) contradicts the claim.
- If the step is real, it must be shown from a trace that is committed, complete and
  tied to its S value; if it is not real, the model's constant factor is not the defect.

**`gpu-sim-ggh` (model correction: size-dependent phase-fusion factor) must not
proceed** until the reference is re-measured. Correcting the model to match a reduction
that its own evidence contradicts would bake an artifact into the model — the exact
error this lab has spent six beads learning to catch, now one level up: the instrument
was not the timing probe but the *ground-truth table*.

## What is still true from 4yw

- Leg 1's pre-registration, harness and rule are fine and reusable.
- Leg 2 remains DESCRIPTIVE and inherits a broken reference, not a broken model.
- The independence claim in Leg 2 (trichotomy and the `n≥3` refusal are structural and do
  not depend on the wire byte factor) is unaffected: those results are test-enforced,
  not table-derived.
