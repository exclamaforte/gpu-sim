# gpu-sim-u2o.17.5 — canonical real-GPU PyTorch trace fixture

24 cases across **5 op classes**, captured with `torch.profiler` (CPU+CUDA
activities, `record_shapes=True`) on the real RTX 5090, plus the raw per-case
chrome traces so the fixture can be re-derived rather than trusted. This bead
produces the fixture only; **17.4 owns pass/fail**, 17.3 fits against this.

Extends the 17.2 approach rather than inventing a harness: same profiler, same
leaf-op mapping (relu → `aten::clamp_min`, matmul → `aten::mm`).

## Op classes and measured spread

| class | cases | median spread (p05–p95 of median) | max spread |
|---|---|---|---|
| elementwise (relu/sigmoid/tanh, 2^10…2^24) | 12 | 5.7% | 44.4% |
| matmul (512…8192, fp32) | 5 | 8.4% | 36.5% |
| reduce (sum/max, 2^14…2^24) | 4 | 2.6% | 4.5% |
| chain (relu→mul→add) | 2 | 5.5% | 7.3% |
| launch_bound (relu, 2^8) | 1 | 3.3% | 3.3% |

Spread is the **p05–p05…p95 band of the dominant kernel's device-event
durations**, not min..max. The first version of this script reported matmul
spreads of 155–230% because it pooled *all* device events in a case's trace,
mixing a first-call cuBLAS init/memset against the steady-state kernel; the
full min..max range is still recorded per case as
`device_ms_full_range_pct_of_median` so nothing is hidden, but it is not the
parity-relevant number.

## Cases that must NOT be used as tight parity targets

1. **`elementwise relu [1024]`, 44.4% spread.** A 0.9 µs measurement. Per
   gpu-sim-u2o.6.6.1, ~1.3 µs of that is irreducible dispatch and another
   ~0.4–1.4 µs is single-launch measurement overhead, so this row is measuring
   the launch path. It is in the fixture deliberately, labelled `launch_bound`
   in spirit and annotated in the case's `note`, so 17.3 does not fit a
   bandwidth model to it.
2. **`elementwise relu [16777216]`, 29.0% spread** — the largest and noisiest
   case, and the telemetry says why: see below.
3. **`matmul [512,512,512]`, 36.5% spread** (10.8 µs, `cutlass_80_simt_sgemm`).
   Small-GEMM launch and tile-quantisation effects. 4096³ is 1.3% and 8192³ is
   9.4%; the mid-to-large matmuls are the usable parity targets.

## Machine state — read this before trusting a parity failure

Telemetry before: clocks 2415 MHz, 44 °C, throttle reasons **0x0**.
Telemetry after: clocks 2197 MHz, 49 °C, throttle reasons
**0x0000000000000004 = SW Power Cap**, persistence Disabled.

So the capture ran into the power cap partway through. That is the honest
machine state of this fixture, it is recorded inside the fixture itself, and
it is the most likely reason the 2^24 elementwise row has a 29% spread. If a
later parity miss is attributed to the model, this attribute should be ruled
out first — which is exactly why the acceptance asked for telemetry inside the
capture.

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
