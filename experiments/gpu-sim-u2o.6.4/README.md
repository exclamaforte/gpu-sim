# gpu-sim-u2o.6.4 — the READ-ONLY half only; the mutating half is on hold

**The hold is part of the acceptance, not a caveat.** `nvidia-smi -lgc` / `-pl`
and persistence-mode setup require root and change **global GPU state shared
with cclm, opguard and chessvar**, which are not this lab's to perturb. That
half needs a human decision and nothing here touches it. A test greps the
executable body for those flags, because a boundary held only in prose is a
boundary that erodes.

## What this does

`gpu_telemetry.py` samples clocks, temperature, power, the individual
**throttle-reason cause bits**, and persistence mode — **read-only**, via
`nvidia-smi --query-gpu`. It performs no device work and cannot contaminate
another tenant's measurement.

Every record it writes carries: *"This is READ-ONLY DIAGNOSTIC telemetry
sampled outside the queue. It is NOT a measurement and must never be cited as
one."* That is item 3, and stamping it onto the artifact is what stops a
telemetry sample being quoted later as evidence about kernel runtime.

## Item 4, answered from measurement: throttling is NOT boost-state

This run:

| | |
|---|---|
| `clocks.sm` relative spread | **0.893** |
| throttle masks observed | **[0]** — never non-zero |
| samples | 12 |
| `clocks.max.sm` | 3105 MHz |
| temperature / power | 45 °C / 86.9 W |
| persistence mode | Disabled |

**The clock moved by 89% with every throttle bit clear.** That is the
distinction item 4 exists to prevent being lost: a throttle bitmask is a set of
**causes**; clock frequency is a **consequence**. This GPU idles low and boosts,
so `clocks.sm` is a boost-state signal and says nothing about throttling on its
own.

The 2026-09-28 contention run made the opposite error — throttle reasons 0 → 4
alongside a `clocks.sm` relative spread of 0.098, which says nothing about
whether timing was affected. The tool records the cause bits and the clock
separately, and reports `throttle_ever_active` on its own so the two cannot be
merged by a later reader.

## A flag I removed from my own tool

The first draft had `--around <command>`, which ran a command beside the
sampling. I used it, and it is exactly the wrong affordance: it invites running
GPU work **outside the queue**, which the rules forbid on a shared box and which
would contaminate another tenant's measurement. It is gone, and
`test_the_tool_runs_nothing` asserts both `--around` and `Popen` are absent from
the executable body.

Sampling *around* a real measurement means the measurement is **in the queue** and
records its own telemetry either side — as `gpu-sim-u2o.6.6.5` does. It does not
mean firing workloads from a diagnostic tool.

## Artifacts

CPU/diagnostic only. No GPU work, no `ts` job.

```
experiments/gpu-sim-u2o.6.4/gpu_telemetry.py   read-only sampler
experiments/gpu-sim-u2o.6.4/{telemetry.json,output.log,job.json}
tests/test_gpu_telemetry.py                     8 tests
```

## Still owed, and not mine to do

Sampling around an actual queued measurement, and any answer to whether
throttling is a factor *during* one. Both need the measurement to be in the
queue; neither can be produced by a diagnostic tool run beside an idle GPU.
