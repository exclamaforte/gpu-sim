# gpu-sim-u2o.12.1 — REAL training step-time / MFU baseline

The audit reopened this bead because `training_throughput_profile.py`'s own
metadata said *"CPU smoke; rerun on 5090 for the real baseline"* — and it never
did. This is the real run: ts `20260928t094902-9a8684a2`,
device `NVIDIA GeForce RTX 5090` (170 SMs, capability
12.0), torch 2.14.0+cu130 / CUDA 13.0.

## Four defects that each produced a plausible wrong answer on the GPU

1. **No `torch.cuda.synchronize()`.** The old loop timed `time.time()` around
   launches without synchronising, so on a GPU it would have measured the time
   to *queue* the work, not to do it. Every step would have looked like a few
   Python calls.
2. **The peak was the wrong GPU.** MFU was divided by **82.6 TFLOP/s — the
   RTX 4090's** FP32 peak. The 5090's is higher, so every MFU would have been
   overstated by the ratio of the two.
3. **The model input never matched the feature vector.** `create_model` defaults
   `numeric_dim=26`; the real vector is **27** on master and **28** with
   gpu-sim-u2o.8.3. Feeding the real vector **raises** (`mat1 and mat2 shapes
   cannot be multiplied (2x28 and 26x128)`). The old script fed
   `torch.randn(batch, 26)` — noise of the wrong width — so it was timing a model
   that cannot consume the actual dataset at all.
4. **`mean, log_var = model(...)` unpacked two values, but the model returns a
   single `(batch, 1)` tensor.** A `(2, 1)` tensor unpacks into two elements, so
   this **worked at batch=2 and raised at every larger batch**. The old script
   could therefore only ever have profiled the smoke rung. Found by a CPU
   preflight over all four rungs *before* spending a queue slot.

Also: `sys.path.insert(0, '/home/gabe/Documents/gpu-sim')` — a hardcoded
machine path, the same class as 8dw's `/tmp` dependency. Now derived from
`__file__`.

## The measured ladder

| rung | params | d_model | layers | batch×seq | median step | achieved TFLOP/s | MFU | training can consume |
|---|---|---|---|---|---|---|---|---|
| `smoke-as-written` | 94,817 | 32 | 1 | 2×16 | 1.516 ms | 0.012 | 0.01% | 1,319 |
| `small` | 581,761 | 128 | 2 | 32×128 | 2.194 ms | 6.515 | 6.22% | 14,583 |
| `medium` | 13,926,785 | 512 | 4 | 32×256 | 19.259 ms | 35.543 | 33.92% | 1,662 |
| `large` | 47,220,609 | 768 | 6 | 16×512 | 68.774 ms | 33.748 | 32.20% | 233 |

`achieved TFLOP/s` is measured; MFU is against a **literature** peak
(104.8 TFLOP/s, literature value for GB202 FP32, NOT measured in this run; treat MFU as carrying the peak's uncertainty), so
MFU inherits that peak's uncertainty — use the achieved column if you distrust
it. The FLOP count is the standard `6·N·seq·batch` approximation, which for a
model with a numeric-feature MLP is an *upper* bound, so these MFU figures are
*lower* bounds.

Note the smoke rung: **MFU 0.01%**. That is what the old CPU-smoke script was
representing, and it is why "we have a training profile" was not the same as
"we have a baseline".

## "The v1 model size" is not defined anywhere

`v1_scope.md` describes the *prediction* scope (single kernel launch), not a
model size. So this profiles a **ladder** and reports the sizes rather than
picking one and calling it v1. Picking a size and labelling it v1 would have
been the same class of error as the rest: a plausible value with no source.

## The answer to gpu-sim-u2o.12.2

DATA GENERATION IS THE BOTTLENECK, decisively. Training can consume 14583 ex/s (small), 1662 ex/s (medium), 233 ex/s (large), while generation PRODUCES 2.39 ex/s. Generation is 97x to 6105x SLOWER than the training side can consume. No training-side optimisation can change that ratio, because the constraint is how fast examples are produced, not how fast they are consumed. gpu-sim-u2o.6.7's own split makes the cause concrete: 99.998% of the generation wall clock is per-example process overhead, not GPU work.

Generation is measured at **2.39 ex/s**
(8,599/hour) from 264 real measured rows
(ts 20260928t074004-b429f469), read from 6.7's committed artifact rather than
retyped, so the comparison cannot drift from its source.

**Caveat, stated:** this compares *throughput*, not end-to-end wall clock for a
whole run, and the two sides were measured in separate jobs. It is the right
comparison for the question 12.2 asks — whether training-speed work could pay —
and not a claim about total project wall clock.

## One arithmetic error of mine, in the summary

The first version of the answer string divided generation by training and
printed **"Generation is 0x to 0x slower"**, because generation is the smaller
number and the numerator was on the wrong side. Fixed to training ÷ generation:
97× to 6105×. Same ratio-direction slip as 6.6.4's roofline and 6.10's
seconds-vs-milliseconds — the error is always *which side is the numerator*, and
it is always in a sentence written by hand rather than computed from the fields.
