# gpu-sim-u2o.17.5 — canonical real-GPU PyTorch trace fixture

**28 cases across 5 op classes, 28 measured, 0 hollow**, captured with
`torch.profiler` (CPU+CUDA activities, `record_shapes=True`) on the real
RTX 5090, plus the raw per-case chrome traces so the fixture can be re-derived
rather than trusted. This bead
produces the fixture only; **17.4 owns pass/fail**, 17.3 fits against this.

Extends the 17.2 approach rather than inventing a harness: same profiler, same
leaf-op mapping (relu → `aten::clamp_min`, matmul → `aten::mm`).

## Op classes and measured spread

All figures below are regenerated from the committed fixture (v3).

| class | cases | median spread (p05–p95 of median) | max spread |
|---|---|---|---|
| elementwise (relu/sigmoid/tanh, 2^10…2^24) | 12 | 6.8% | 53.8% |
| matmul (512…8192, fp32) | 5 | 9.9% | 35.9% |
| reduce (sum/max, 2^14…2^24) | **8** | 2.2% | 4.7% |
| chain (relu→mul→add) | 2 | 5.5% | 7.2% |
| launch_bound (relu, 2^8) | 1 | 3.2% | 3.2% |

Spread is the **p05–p05…p95 band of the dominant kernel's device-event
durations**, not min..max. The first version of this script reported matmul
spreads of 155–230% because it pooled *all* device events in a case's trace,
mixing a first-call cuBLAS init/memset against the steady-state kernel; the
full min..max range is still recorded per case as
`device_ms_full_range_pct_of_median` so nothing is hidden, but it is not the
parity-relevant number.

## Cases that must NOT be used as tight parity targets

1. **`elementwise relu [1024]`, 53.8% spread.** A 0.87 µs measurement. Per
   gpu-sim-u2o.6.6.1, ~1.3 µs of that is irreducible dispatch and another
   ~0.4–1.4 µs is single-launch measurement overhead, so this row is measuring
   the launch path. It is in the fixture deliberately, labelled `launch_bound`
   in spirit and annotated in the case's `note`, so 17.3 does not fit a
   bandwidth model to it.
2. **`elementwise relu [65536]`, 18.5%** and **`relu [16777216]`, 25.3%** — also
   launch-path effects at the small end and power-cap noise at the large end.
3. **`matmul [512,512,512]`, 35.9% spread** (10.8 µs, `cutlass_80_simt_sgemm`).
   Small-GEMM launch and tile-quantisation effects; 4096³ is the clean one.

## Per-case machine state — the reviewer asked for this, and it paid off

Every case now carries its own telemetry and a `throttle_active` flag.
**11 of 28 cases were captured while the SW power cap was active**, including
the 8192³ matmul and *both* chain cases — whose spreads (5.5% and 7.2%) look
perfectly acceptable. Spread alone would not have flagged them. The
before/after pair could only ever have proved that throttling happened
somewhere in the capture; the per-case stamp localises it, and 17.4 can now
exclude contaminated cases mechanically instead of by reading a bd comment.

The four exclusions above are the cases to avoid on spread grounds; the 11
throttle-flagged cases are the ones to exclude on machine-state grounds. They
overlap only once.

## Machine state, capture-level

Telemetry before: clocks 195 MHz, 37 °C, throttle reasons **0x0**.
Telemetry after: clocks 2197 MHz, 49 °C, throttle reasons
**0x0000000000000004 = SW Power Cap**, persistence Disabled.

So the capture ran into the power cap partway through. Recorded inside the
fixture, and localised per case above. If a later parity miss is attributed to
the model, this attribute should be ruled out first.

## The four hollow cases, and what now prevents them

The v2 revision shipped 8 `reduce` entries of which 4 were **empty dicts** —
`torch.max` dispatches to `aten::max`, I looked for `aten::amax`, found
nothing, and nothing in the fixture or the counts disclosed it. The summary
counted only the populated half, which is exactly how it slipped through.

Fixed at the source (the leaf-op mapping) and guarded structurally: the
capture now **rejects the whole fixture if any case has no
`device_ms_median`**, printing `HOLLOW <class>/<op> <shapes>` and exiting
non-zero. v3 reports `28 measured cases ... 0 hollow`.

## Negative control

`capture_fixture.py --self-test` with `CUDA_VISIBLE_DEVICES=` exits **1**
(`RuntimeError: No CUDA GPUs are available`) and the run refuses to write a
fixture if the control does not fail. The result is stored in the fixture at
`protocol.negative_control`.

## Files

* `capture_fixture.py` — the capture; `run.sh` — ts entry point
* `canonical_fixture.json` — the fixture 17.3 consumes (committed)
* `traces/case_*.json` — raw chrome trace per case, so every number is
  re-derivable
* `output.log`, `job.json`
