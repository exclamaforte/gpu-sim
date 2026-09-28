#!/usr/bin/env python3
"""Measure the noise floor of the real measurement path (gpu-sim-u2o.6.6.1).

Run ONLY through the ts queue (gpu-sim-u2o.6.8). One short job, fixed
reference kernel, read-only.

Deliberate constraints honoured here:
  * preflight_check is BYPASSED. Per the PI's ruling on 6.1.1 it currently
    executes on the real GPU; inheriting an unaccounted second execution is
    exactly what this experiment must not do, so we call compile_kernel() and
    run_kernel() directly. Stated again in output.log.
  * fixed hand-written reference kernel; nothing generated or untrusted.
  * no sudo, no nvidia-smi -lgc, no persistence-mode change. Telemetry is
    read-only queries only.
  * three INDEPENDENT process invocations per cell, because the variance of
    the reported median across invocations is the quantity that matters --
    within-process cv alone understates it.
"""
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from epic_u2o.safe_harness import SafeHarness  # noqa: E402

# ---- design -----------------------------------------------------------------
BLOCK = 256
TRIALS = 200          # large enough to characterise the distribution
BATCH = 100           # launches per timed region in the batched mode
INVOCATIONS = 3       # independent processes per cell
# two decades of problem size, so launch-bound and compute-bound are both seen
SIZES = [1 << 10, 1 << 14, 1 << 18, 1 << 22, 1 << 26]
MDD_TRIAL_COUNTS = [10, 20, 50, 100, 200]

TELEMETRY_FIELDS = [
    "clocks.sm", "clocks.max.sm", "temperature.gpu",
    "clocks_throttle_reasons.active", "persistence_mode",
    "power.draw", "power.limit", "utilization.gpu", "memory.used",
]


def telemetry(label: str) -> dict:
    """Read-only nvidia-smi query. No lock, no reset, no persistence change."""
    out = {"label": label, "at": time.time()}
    try:
        r = subprocess.run(
            ["nvidia-smi", f"--query-gpu={','.join(TELEMETRY_FIELDS)}",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30)
        out["rc"] = r.returncode
        out["csv"] = r.stdout.strip()
        if r.returncode == 0 and r.stdout.strip():
            out["fields"] = dict(zip(TELEMETRY_FIELDS,
                                     [v.strip() for v in r.stdout.strip().split(",")]))
    except Exception as e:
        out["error"] = str(e)
    return out


def negative_control(exe) -> dict:
    """Run the SAME binary with no visible CUDA device; it must fail.

    Acceptance criterion 4. A measurement that silently ran on nothing looks
    exactly like a measurement of a very fast kernel, so this control is what
    tells the two apart. The positive control (same invocation WITH the device)
    is recorded too, so the failure is attributable to the missing device
    rather than to the binary.
    """
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    r = subprocess.run([str(exe), "scale", "4096", "256", "5", "1"],
                       capture_output=True, text=True, timeout=120, env=env)
    r2 = subprocess.run([str(exe), "scale", "4096", "256", "5", "1"],
                        capture_output=True, text=True, timeout=120)
    return {
        "what": "same binary, CUDA_VISIBLE_DEVICES= blanked",
        "returncode": r.returncode,
        "stdout_tail": r.stdout.strip()[-200:],
        "stderr_tail": r.stderr.strip()[-200:],
        "failed_as_required": r.returncode != 0,
        "positive_control_returncode": r2.returncode,
        "positive_control_succeeded": r2.returncode == 0,
    }


def parse_trials(stdout: str) -> tuple[list[float], dict]:
    trials, summary, bad = [], {}, []
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            # Keep going, but report it: one unparseable line must not throw
            # away 200 good trials, and it must not be silently swallowed
            # either -- an interleaved write is exactly the kind of thing
            # that would bias a noise-floor measurement.
            bad.append(line[:200])
            continue
        if rec.get("summary"):
            summary = rec
        else:
            trials.append(float(rec["us"]))
    return trials, summary, bad


def stats_of(trials: list[float]) -> dict:
    s = sorted(trials)
    # Keep the raw vector (reviewer MINOR on 6.6.1): with only median/std/cv a
    # third party cannot re-derive other percentiles, or check for outliers and
    # trimming. 200 floats per cell is cheap and makes every statistic in this
    # file reproducible from the artifact alone.
    raw = [round(x, 6) for x in trials]
    n = len(s)
    q1, q3 = s[n // 4], s[(3 * n) // 4]
    med = statistics.median(s)
    mean = statistics.fmean(s)
    sd = statistics.stdev(s) if n > 1 else 0.0
    return {
        "n_trials": n,
        "median_ms": med / 1000.0,
        "iqr_ms": (q3 - q1) / 1000.0,
        "mean_ms": mean / 1000.0,
        "std_ms": sd / 1000.0,
        "cv": (sd / mean) if mean else 0.0,
        "median_us": med,
        "iqr_us": q3 - q1,
        "mean_us": mean,
        "std_us": sd,
        "p05_us": s[max(0, int(n * 0.05))],
        "p95_us": s[min(n - 1, int(n * 0.95))],
        "raw_us": raw,
    }


def mdd_pct_by_n(stats: dict, n_list=MDD_TRIAL_COUNTS) -> dict:
    """Minimum detectable RELATIVE difference at 2 sigma, per trial count.

    Comparison of two independently measured medians, each from n trials:
        SE(median) ~= 1.2533 * sigma / sqrt(n)      (asymptotic normal approx
                                                     for the sample median)
        MDD = z_{2sigma} * sqrt(2) * SE(median)    (two-sample, z=1.96)
        MDD% = 100 * MDD / median
    """
    med, sd = stats["median_us"], stats["std_us"]
    out = {}
    for n in n_list:
        se = 1.2533 * sd / (n ** 0.5)
        out[str(n)] = 100.0 * (1.96 * (2 ** 0.5) * se) / med if med else None
    return out


def main() -> int:
    print("=" * 72)
    print("gpu-sim-u2o.6.6.1 -- noise floor of the real measurement path")
    print("=" * 72)
    print("PREFLIGHT IS BY-PASSED ON PURPOSE: SafeHarness.preflight_check()")
    print("currently executes on the real GPU (PI ruling on 6.1.1), so calling")
    print("it here would add a second, unaccounted GPU execution to a")
    print("measurement experiment. compile_kernel() and run_kernel() are called")
    print("directly instead. Kernel is a FIXED hand-written reference kernel:")
    print("no generated or untrusted code, no dependency on the fake path.")
    print("No sudo, no clock locking, no persistence-mode change; nvidia-smi is")
    print("used only for read-only telemetry queries.")
    print("=" * 72, flush=True)

    tel_before = telemetry("before")
    print("telemetry before: " + json.dumps(tel_before.get("fields", tel_before)),
          flush=True)

    # compile once, outside the harness pipeline, through the harness's own
    # compile_kernel (real nvcc, real sm_120)
    harness = SafeHarness(timeout_sec=120, vram_limit_mb=24576)
    exe = HERE / "reference_kernel.out"
    src = (HERE / "reference_kernel.cu").read_text()
    ok, msg = harness.compile_kernel(src, str(exe), arch="sm_120")
    print(f"compile_kernel -> ok={ok} {msg[:200]}", flush=True)
    if not ok:
        (HERE / "metrics.json").write_text(json.dumps(
            {"bead": "gpu-sim-u2o.6.6.1", "error": f"compile failed: {msg[:2000]}"},
            indent=2))
        return 1

    # ---- NEGATIVE CONTROL (acceptance criterion 4) ------------------------
    # The same binary, with no CUDA device visible, MUST fail. Without this,
    # "it measured the real GPU" is an assumption; with it, a fake or a
    # silently-empty run would show up as a successful control.
    nc = negative_control(exe)
    print("negative control (CUDA_VISIBLE_DEVICES=): "
          f"failed_as_required={nc['failed_as_required']} "
          f"rc={nc['returncode']} err={nc['stderr_tail'][:120]!r}", flush=True)

    cells = []
    t0 = time.time()
    for kernel in ("empty", "scale"):
        for n in SIZES:
            for mode, batch in (("single", 1), ("batched", BATCH)):
                per_inv = []
                for inv in range(INVOCATIONS):
                    res = harness.run_kernel(
                        str(exe), args=[kernel, str(n), str(BLOCK),
                                        str(TRIALS), str(batch)])
                    if not res.success:
                        print(f"FAILED {kernel} n={n} {mode} inv{inv}: "
                              f"rc={res.returncode} {res.stderr[:200]!r}", flush=True)
                        continue
                    trials, summary, bad = parse_trials(res.stdout)
                    if bad:
                        # Never silently swallow a bad line: an interleaved or
                        # truncated write is exactly the kind of thing that
                        # would bias a noise-floor measurement.
                        print(f"  !! {kernel} n={n} {mode} inv{inv}: "
                              f"{len(bad)} unparseable stdout line(s); first: "
                              f"{bad[0]!r}", flush=True)
                        if inv == 0:
                            (HERE / f"raw_stdout_{kernel}_{n}_{mode}.txt").write_text(
                                res.stdout)
                            print(f"     raw stdout saved to raw_stdout_{kernel}_{n}_{mode}.txt",
                                  flush=True)
                    st = stats_of(trials)
                    st["invocation"] = inv
                    st["grid"] = summary.get("grid")
                    per_inv.append(st)
                    print(f"  {kernel:5s} n={n:<9d} {mode:6s} inv{inv}  "
                          f"trials={st['n_trials']:<4d} median={st['median_us']:.4f}us "
                          f"iqr={st['iqr_us']:.4f}us cv={st['cv']:.4f} "
                          f"grid={summary.get('grid')}", flush=True)
                if not per_inv:
                    continue
                meds = [s["median_us"] for s in per_inv]
                cell = {
                    "kernel": kernel,
                    "n_problem_size": n,
                    "grid": per_inv[0]["grid"],
                    "block": BLOCK,
                    "mode": mode,
                    "batch": batch,
                    "n_trials": TRIALS,
                    "n_invocations": len(per_inv),
                    "median_ms": statistics.median(meds) / 1000.0,
                    "iqr_ms": statistics.median([s["iqr_us"] for s in per_inv]) / 1000.0,
                    "mean_ms": statistics.median([s["mean_us"] for s in per_inv]) / 1000.0,
                    "std_ms": statistics.median([s["std_us"] for s in per_inv]) / 1000.0,
                    "cv": statistics.median([s["cv"] for s in per_inv]),
                    "median_across_invocations_ms": [m / 1000.0 for m in meds],
                    "median_spread_pct": (100.0 * (max(meds) - min(meds))
                                          / statistics.median(meds)) if meds else None,
                    "across_invocations_spread_pct": (
                        100.0 * (max(meds) - min(meds)) / statistics.median(meds)
                        if meds else None),
                    "per_invocation": per_inv,
                    "mdd_pct_by_n": mdd_pct_by_n(per_inv[0]),
                }
                cells.append(cell)
    tel_after = telemetry("after")
    print("telemetry after:  " + json.dumps(tel_after.get("fields", tel_after)),
          flush=True)

    # ---- the headline numbers ------------------------------------------
    by = {(c["kernel"], c["mode"], c["n_problem_size"]): c for c in cells}
    empty_single = by.get(("empty", "single", SIZES[0]))
    empty_batched = by.get(("empty", "batched", SIZES[0]))
    launch_overhead_us = None
    if empty_single and empty_batched:
        launch_overhead_us = (empty_single["median_ms"] - empty_batched["median_ms"]) * 1000.0

    # Crossover, defined from the data rather than asserted: the smallest n
    # where the KERNEL's own cost -- batched scale minus batched empty at the
    # same n, so the launch cost is cancelled out -- becomes significant
    # (>= 10% of that cell's empty-kernel launch cost). Below that point a
    # measurement of this kernel is a measurement of launch overhead.
    crossover = None
    crossover_abs_us = None
    for n in SIZES:
        s_b = by.get(("scale", "batched", n))
        e_b = by.get(("empty", "batched", n))
        if not (s_b and e_b):
            continue
        kernel_us = s_b["median_ms"] * 1000.0 - e_b["median_ms"] * 1000.0
        launch_us = e_b["median_ms"] * 1000.0
        if kernel_us > 0.10 * launch_us:
            crossover, crossover_abs_us = n, kernel_us
            break

    # Same question in absolute terms: where does kernel work exceed 1 us?
    n_above_1us = None
    for n in SIZES:
        s_b = by.get(("scale", "batched", n))
        e_b = by.get(("empty", "batched", n))
        if s_b and e_b and (s_b["median_ms"] - e_b["median_ms"]) * 1000.0 > 1.0:
            n_above_1us = n
            break

    # Per-size decomposition: how much of the reported single-launch number is
    # actually kernel work. This is the headline the PI asked for.
    decomposition = {}
    for n in SIZES:
        s_s = by.get(("scale", "single", n))
        e_s = by.get(("empty", "single", n))
        s_b = by.get(("scale", "batched", n))
        e_b = by.get(("empty", "batched", n))
        if not all((s_s, e_s, s_b, e_b)):
            continue
        total = s_s["median_ms"] * 1000.0
        launch_b = e_b["median_ms"] * 1000.0
        kernel = s_b["median_ms"] * 1000.0 - launch_b
        overhead = (e_s["median_ms"] - e_b["median_ms"]) * 1000.0
        decomposition[str(n)] = {
            "components_measured_in_different_modes": (
                "NOT a partition: reported_single is single-mode scale; "
                "irreducible_launch and kernel_work are batched-mode "
                "differences; measurement_overhead is single-minus-batched. "
                "The residual column is what is left over, and it is not "
                "small -- do not sum these four as if they decomposed "
                "exactly. The kernel_work figure itself is sound: it is the "
                "internally consistent batched difference."),
            "reported_single_us": total,
            "irreducible_launch_us": launch_b,
            "measurement_overhead_us": overhead,
            "kernel_work_us": kernel,
            "residual_us": total - (launch_b + overhead + kernel),
            "residual_pct_of_reported": (
                100.0 * (total - (launch_b + overhead + kernel)) / total
                if total else None),
            "kernel_fraction_of_reported": (kernel / total) if total else None,
        }

    # trials needed to resolve a 10% difference, per cell
    trials_for_10pct = {}
    for c in cells:
        if c["kernel"] != "scale" or c["mode"] != "single":
            continue
        sd, med = c["per_invocation"][0]["std_us"], c["per_invocation"][0]["median_us"]
        se10 = 0.10 * med / (1.96 * (2 ** 0.5))
        n_need = int(round((1.2533 * sd / se10) ** 2)) if se10 > 0 else None
        trials_for_10pct[str(c["n_problem_size"])] = n_need

    # Same question for the BATCHED timing mode, which is the actionable one:
    # it is what the measurement design should switch to if the single-launch
    # numbers turn out to be launch-dominated.
    trials_for_10pct_batched = {}
    for c in cells:
        if c["kernel"] != "scale" or c["mode"] != "batched":
            continue
        sd, med = c["per_invocation"][0]["std_us"], c["per_invocation"][0]["median_us"]
        se10 = 0.10 * med / (1.96 * (2 ** 0.5))
        n_need = int(round((1.2533 * sd / se10) ** 2)) if se10 > 0 else None
        trials_for_10pct_batched[str(c["n_problem_size"])] = n_need

    # Across-invocation spread of the median, per cell. This is the range the
    # reviewer asked to see stated accurately: within-process cv is NOT it.
    aicv = [c["across_invocations_spread_pct"] for c in cells
            if c.get("across_invocations_spread_pct") is not None]
    aicv_batched = [c["across_invocations_spread_pct"] for c in cells
                    if c["mode"] == "batched"
                    and c.get("across_invocations_spread_pct") is not None]
    aicv_single = [c["across_invocations_spread_pct"] for c in cells
                   if c["mode"] == "single"
                   and c.get("across_invocations_spread_pct") is not None]

    noise_floor = {
        "negative_control": nc,
        "across_invocation_spread_pct": {
            "all_cells_min": min(aicv) if aicv else None,
            "all_cells_max": max(aicv) if aicv else None,
            "single_mode_max": max(aicv_single) if aicv_single else None,
            "batched_mode_min": min(aicv_batched) if aicv_batched else None,
            "batched_mode_max": max(aicv_batched) if aicv_batched else None,
            "note": ("spread of the reported MEDIAN across the 3 independent "
                     "process invocations, as a percentage of that median. "
                     "This, not the within-process cv, is what limits "
                     "reproducibility."),
        },
        "empty_kernel_us": empty_single["median_ms"] * 1000.0 if empty_single else None,
        "empty_kernel_batched_us": (empty_batched["median_ms"] * 1000.0
                                    if empty_batched else None),
        "grid_cap_note": ("grid.x is capped at 65535 (classic launch bound), so at "
                          "n > 65535*block = 16777216 the kernel only covers "
                          "grid*block elements. The n=2^26 cell therefore measures "
                          "the first 16777216 elements, not 67108864."),
        "launch_overhead_us_single_minus_batched": launch_overhead_us,
        "launch_bound_crossover_n": crossover,
        "crossover_definition": ("smallest n where (batched scale - batched empty) "
                                "exceeds 10% of that cell's empty-kernel launch "
                                "cost; below it, measuring this kernel measures "
                                "launch overhead"),
        "crossover_kernel_cost_us": crossover_abs_us,
        "n_where_kernel_work_exceeds_1us": n_above_1us,
        "reported_time_decomposition": decomposition,
        "mdd_pct_by_n": empty_single["mdd_pct_by_n"] if empty_single else {},
        "mdd_pct_by_n_note": ("MDD is computed on the EMPTY kernel cell (the "
                              "worst case: the smallest, most overhead-dominated "
                              "signal on this box). Per-cell MDD is in each "
                              "cell's mdd_pct_by_n."),
        "trials_needed_for_10pct_difference": trials_for_10pct,
        "trials_needed_for_10pct_difference_batched": trials_for_10pct_batched,
    }

    metrics = {
        "bead": "gpu-sim-u2o.6.6.1",
        "what": "noise floor of the real 5090 measurement path",
        "preflight_bypassed": True,
        "preflight_bypass_reason": ("preflight_check executes on the real GPU "
                                    "(PI ruling 6.1.1); calling it would add an "
                                    "unaccounted second GPU execution"),
        "telemetry_before": tel_before,
        "telemetry_after": tel_after,
        "design": {"block": BLOCK, "trials": TRIALS, "batch": BATCH,
                   "invocations": INVOCATIONS, "sizes": SIZES},
        "noise_floor": noise_floor,
        "cells": cells,
        "total_wall_sec": time.time() - t0,
    }
    # FAIL-CLOSED stamp guard (PI ruling, shared with 6.1.1). A guard that
    # fails OPEN is worse than none, because it looks like protection: a
    # check wrapped in `if baseline.exists()` is skipped exactly when the
    # stamped artifact it protects is missing. So refuse to write, loudly,
    # if the baseline this run would supersede is absent or mis-stamped, and
    # write the successor to its OWN file so the baseline is never at risk.
    baseline = HERE / "metrics_v1.json"
    if not baseline.exists():
        print("FATAL: refusing to write. metrics_v1.json (noise_floor_v1, ts "
              "20260927t162535-cb00ea78) is absent, so this run has no "
              "declared predecessor to supersede. Restore it, or re-stamp it "
              "deliberately as v2 with a stated reason. This guard is "
              "fail-CLOSED on purpose.", flush=True)
        return 2
    try:
        bv1 = json.loads(baseline.read_text())
    except Exception as e:
        print(f"FATAL: refusing to write. metrics_v1.json is unreadable: {e}",
              flush=True)
        return 2
    if bv1.get("measurement_version") != 1:
        print(f"FATAL: refusing to write. metrics_v1.json declares "
              f"measurement_version {bv1.get('measurement_version')!r}, not 1.",
              flush=True)
        return 2
    (HERE / "metrics_v2.json").write_text(json.dumps(metrics, indent=2))
    print(f"wrote metrics_v2.json; the noise_floor_v1 baseline at "
          f"{baseline.name} is untouched", flush=True)
    (HERE / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("\n=== NOISE FLOOR ===")
    print(json.dumps(noise_floor, indent=2))
    print(f"\ntotal wall {time.time() - t0:.1f}s, {len(cells)} cells")
    return 0 if cells else 1


if __name__ == "__main__":
    raise SystemExit(main())
