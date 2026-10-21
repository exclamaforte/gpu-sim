# gpu-sim-4yw: Validate model against measured pieces-per-phase table

## RULING UPDATE (PI review, commit 7bb3764): REFERENCE-UNVERIFIED

The Leg 1 verdict below (`MODEL-DOES-NOT-REPRODUCE`) is **superseded**. The PI
reviewed the reference table (`canonical_byte_table.md`) against its own cited
traces and found that the table disagrees with its traces on every row. The
specific claim Leg 1 rests on — a 2.0→1.0 byte-factor step at ≥512 KiB — is
absent from the raw data: at 256 KiB and 512 KiB the trace shape is identical
(16 sends/rank, S/4 bytes/send, 4S total). The correct verdict is
`REFERENCE-UNVERIFIED`, not `MODEL-DOES-NOT-REPRODUCE`.

The comparison *method* is sound and the pre-registration was fixed in advance
(verifiable from the commit graph: pre-reg 65cb113 at 08:42:02, results
463d6cb at 08:42:53, 51 seconds apart). The *reference* is unusable.

Consequences:
- `gpu-sim-ggh` (model correction) must NOT proceed until the reference is
  re-measured.
- The model must NOT be corrected to fit a table that its own evidence
  contradicts.
- Leg 2 remains DESCRIPTIVE and inherits a broken reference, not a broken model.
- The independence claim (trichotomy and n≥3 refusal are structural) is
  unaffected.

Full analysis: `PI_check_reference_table.md` (preserved traces in
`stale_traces_from_tmp/`, byte-identical to `/tmp/scratchpad/` originals).

---

## Leg 1 — FALSIFIABLE: MODEL-DOES-NOT-REPRODUCE

Verdict (fixed rule, 10% relative on per-phase wire bytes): **MODEL-DOES-NOT-REPRODUCE**
(3 of 6 phases outside tolerance).

| S (bytes) | measured P | meas per-phase bytes | model per-phase bytes | rel err | within 10%? |
|-----------|-----------:|--------------------:|----------------------:|--------:|:-----------:|
| 4096      | 1          | 4096 (S)            | 4096                  | +0.000  | yes |
| 65536     | 4          | 65536 (S)           | 65536                 | +0.000  | yes |
| 262144    | 4          | 262144 (S)          | 262144                | +0.000  | yes |
| 524288    | 8          | 262144 (S/2)        | 524288                | +1.000  | **no** |
| 1048576   | 8          | 524288 (S/2)        | 1048576               | +1.000  | **no** |
| 2097152   | 8          | 1048576 (S/2)       | 2097152               | +1.000  | **no** |

Inputs (all from committed artifacts):
- Fabric params: bw=10.0 Gbps, lat=5.0 us, loss=0, jitter=0 (experiments/gpu-sim-pkb/run.sh single-link topo).
- Piece counts P: experiments/gpu-sim-pkb/canonical_byte_table.md (override=True in predict()).
- Measured per-phase bytes: experiments/gpu-sim-pkb/canonical_byte_table.md.

Failing phases: S = 524288, 1048576, 2097152 bytes (>= 512 KiB).
Direction and size of error: model over-predicts per-phase wire bytes by 2.0x
(+100% relative) at these sizes. The model assumes a constant byte factor of 2.0
(2 phases x S), but the measurement shows the two phases fuse (byte factor 1.0,
i.e. S/2 per phase) at >= 512 KiB.

Likely missing term: a size-dependent phase-fusion / byte-factor step. The
model's `_factor`/`_bw_factor` for ring_allreduce returns a constant 2.0 for
n=2, but the measured table shows the factor steps from 2.0 to 1.0 between
256 KiB and 512 KiB. The model has no representation of this threshold (it
only varies pieces_per_phase P, not the phase count/byte factor). See
canonical_byte_table.md "Note: the byte factor changes from 2 to 1...".

Secondary (wall-clock duration, NOT the gate): measured medians 6830 us (4 KiB)
and 6729 us (2 MiB) vs model 17 us and 3435 us; the measured values carry
~46 ms scheduler spread on this box (variance_measurement.md) and are unusable
as a gate, hence the tolerance is on the deterministic wire bytes.

No model adjustment was made in this bead. Model-correction bead to be filed
separately.

## Leg 2 — DESCRIPTIVE (model extrapolation; inherits Leg 1 failure)

**This table is DESCRIPTIVE, not a measurement.** Because Leg 1 failed, every
value below inherits that failure (the model's byte factor is wrong at >= 512 KiB);
the table is provided only to map where the categorical conclusions would flip
*if* the model were corrected, and must not be cited as a validated prediction.

Envelope: bw {25,50,100,200,400} Gbps x lat {2,5,10,20} us x loss {0,100,1000} ppm,
at S=2 MiB, n=2, ring_allreduce, P=8. Full table: leg2_results.json (60 points).

Dominant-term counts across the envelope: bandwidth 51/60, latency 9/60, loss 0/60.
Loss never dominates (1000 ppm -> loss_factor 1.001, a 0.1% extra on the wire term).
Latency dominates only at the low-bandwidth / high-latency corner (25 Gbps, 20 us).

Boundary statement (conditional on a corrected byte factor):
- Region-insensitive: the fault trichotomy (loss vs jitter vs timeout) and the
  n>=3 refusal are structural and do not depend on the wire byte factor.
- Region-dependent: any conclusion that hinges on the absolute per-phase cost
  or on the bandwidth/latency crossover (e.g. "is this fabric bandwidth-bound?")
  depends on the byte factor and is unreliable until the model is corrected.
- The 1000 ppm loss point does not change the dominant term anywhere in the
  envelope (loss is a second-order effect at these rates).

## Validity domain

Leg 1 established that the model fails to reproduce the measured wire bytes at
>= 512 KiB (factor 2 over-prediction). Therefore Leg 2's values are ungrounded
extrapolations and inherit that failure. The model is NOT grounded on the wire
axis; only the loss axis (gpu-sim-4tw) is validated.
EOF
echo written