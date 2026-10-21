# gpu-sim-4yw pre-registration — written before any measurement

## 1. Claim under test

`gpu-sim-5zv`'s model route (`epic_cluster/link_model.py::predict`) claims to
predict the wire cost of a collective given fabric parameters (bw, latency,
loss, jitter) and piece counts. `gpu-sim-4tw` validated the loss axis only
(1/(1-p)). This bead validates the central claim: the model's predicted
per-phase wire bytes/cost must reproduce the measured pieces-per-phase table
(`experiments/gpu-sim-pkb/canonical_byte_table.md`, `phase_payload_table.md`).

## 2. Compared phases (all rows of the measured table)

All 6 rows of `experiments/gpu-sim-pkb/canonical_byte_table.md`:

| S (bytes) | floats | measured P (pieces/phase) | measured per-phase bytes |
|-----------|--------|---------------------------|--------------------------|
| 4096      | 1024   | 1                         | 4096 (S)                 |
| 65536     | 16384  | 4                         | 65536 (S)                |
| 262144    | 65536  | 4                         | 262144 (S)               |
| 524288    | 131072 | 8                         | 262144 (S/2)             |
| 1048576   | 262144 | 8                         | 524288 (S/2)             |
| 2097152   | 524288 | 8                         | 1048576 (S/2)            |

Inputs fed to the model (from committed artifacts, not re-derived by hand):
- Fabric params: bw_gbps=10.0, latency_us=5.0, loss_ppm=0, jitter_us=0.0
  (from `experiments/gpu-sim-pkb/run.sh` single-link topo spec).
- n_ranks=2, algorithm=ring_allreduce.
- pieces_per_phase = measured P from the table (override, recorded as override=True).

## 3. Tolerance (fixed, not widened after results)

10% relative on the per-phase WIRE BYTES (the wire term), i.e.
`|pred_per_phase_bytes - meas_per_phase_bytes| / meas_per_phase_bytes <= 0.10`.

Justification (one sentence): the wire term is the model's central claim and is
deterministic from committed artifacts, whereas wall-clock duration on this box is
unusable as a gate (46 ms scheduler spread vs 36 us requirement,
`experiments/gpu-sim-pkb/variance_measurement.md`), so the tolerance is placed on
the deterministic quantity.

## 4. Decision rule (fixed)

- `MODEL-REPRODUCES` iff every one of the 6 phases is within tolerance.
- `MODEL-DOES-NOT-REPRODUCE` otherwise, with failing phases named, the
  direction and size of the error, and the likely missing term.
- Rule is fixed before data; not widened after.

## 5. Leg 2 (descriptive, explicitly labelled)

Sweep: bandwidth {25,50,100,200,400} Gbps x latency {2,5,10,20} us x
loss_ppm {0,100,1000}, at S=2097152 (2 MiB), n=2, ring_allreduce, P=8.
Report dominant term (bandwidth/latency/loss) and predicted per-phase cost per
point, plus a boundary statement. Leg 2 is DESCRIPTIVE (model extrapolation),
not a measurement; its validity domain is whatever Leg 1 established. If Leg 1
fails, Leg 2 values inherit that failure and are not presented as meaningful.

## 6. Reporting obligation

No admission claim. No reopening of timing thread. If Leg 1 fails, file a
separate model-correction bead rather than adjusting the model here.
EOF
echo written