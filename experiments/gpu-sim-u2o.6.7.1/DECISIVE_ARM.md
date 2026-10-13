# The decisive arm: 7.1's isolation, and the offset does not move

This is the run I named as the one that would settle it. Independent **process
per invocation** — 7.1's isolation — with the examples, the 5 warmup + 20 timed
per invocation, the 40 timed launches and the median-of-invocation-medians
estimator all held identical. Only the process boundary changes.
ts `20260928t125506-50c9a9b3`.

| arm | reproduce | median rel diff | rows faster | ai_cv | ex/s |
|---|---|---|---|---|---|
| single-run 7.1 (reference) | — | — | — | 0.1120 | 2.39 |
| batched, one shared context | 206/264 | 12.6% | 70.5% | 0.0344 | 167.30 |
| **batched, 528 independent processes** | **210/264** | **12.5%** | 84.5% | 0.0724 | 3.78 |

**Process isolation is not the cause.** Restoring 7.1's isolation exactly and
the divergence does not collapse — 12.5% against the shared-context arm's 12.6%.

## Four hypotheses now eliminated

| hypothesis | arm | verdict |
|---|---|---|
| shared CUDA context | `reset` | refuted |
| warm state between examples | `cold` (205/264, 11.9%) | refuted |
| the tolerance rule | three like-for-like rules | refuted (12.6% under all) |
| **process isolation** | **this arm** | **refuted** |

## And the smoking gun is a confound I introduced

**7.1 does not measure with a batched driver.** It calls
`epic_u2o.kernel_driver.synthesize()` to build a **separate binary per kernel**
(`k_<hash>.out`) and runs each through `safe_harness.run_kernel()`. My batched
worker builds **one binary** containing all 8 kernels with a dispatch switch, and
allocates four buffers of a shared capacity regardless of the kernel's arity.

So my repeated claim that the protocol is identical **overstated what is held
fixed**. The examples, the launch counts and the estimator are identical; **the
executable is not.** Same isolation, same estimator, different binary — and the
offset persists. That makes the binary the leading candidate, and it is a
difference *I* introduced rather than one the data contains.

## What this means for the 70x, and it is not a small thing

**Process isolation was not worth 70x anyway.** The per-invocation-process arm
runs at **3.78 ex/s** against 7.1's 2.39 — a **1.6x** win, not 70x.

So the 70x depended *entirely* on sharing one context, and if the
shared-context measurements are not comparable to 7.1's, then **the 70x and the
criterion-2 failure are the same fact seen from two sides**: the speed comes from
exactly the change that breaks comparability. That is a much more uncomfortable
position than either number alone, and it is the one the evidence supports.

## Next step

Make the batched path use **7.1's own per-kernel binaries** — call
`kernel_driver.synthesize()` per kernel and dispatch to them from the persistent
process, or simply measure the committed `k_<hash>.out` binaries directly. If the
offset then collapses, the binary was the cause and the fix is to synthesise
7.1's driver rather than my own, which is a small change.

Until then: criterion 2 fails, four hypotheses are eliminated, the binary is
identified as the leading candidate, and under R1 the single-run path remains
authoritative. No re-measurement yet.
