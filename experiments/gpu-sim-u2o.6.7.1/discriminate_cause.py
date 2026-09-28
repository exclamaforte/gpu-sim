#!/usr/bin/env python3
"""Separate the two candidate causes of 6.7.1's criterion-2 failure.

Criterion 2 says a throughput fix that changes the numbers is worse than no fix,
and the batched run fails it: 189/264 medians reproduce, median relative
difference 15.6%. I named TWO candidate causes and said separating them needs a
run that varies one at a time. This is that run.

  H1  the two invocations no longer have independent CUDA contexts, so 7.1's
      ai_cv measured context-to-context variation and this one does not
  H2  warm state carries between EXAMPLES in a persistent process, so the later
      examples run with L2/TLB/allocator state left by the earlier ones

Three runs, protocol IDENTICAL in all of them -- 2 invocations x (5 warmup + 20
timed) = 40 timed launches per example. Only the ISOLATION varies:

  warm   the configuration that failed (baseline to beat)
  cold   buffers freed and reallocated between EXAMPLES        -> tests H2
  reset  cudaDeviceReset between the two INVOCATIONS           -> tests H1

Whichever mode reproduces 7.1's medians has identified the cause. If NEITHER
does, the cause is neither, and that is the finding -- a third hypothesis I have
not thought of is better than confidently attributing it to one of two.

CPU for the comparison; the three runs are GPU work and go through ts.
"""
import json
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

SINGLE_REF = "lab/gpu-sim-u2o.7.1"
MODES = ("warm", "cold", "reset")
TOL_FLOOR, TOL_CAP, TOL_SIGMA = 0.10, 0.75, 3.0


def single_run() -> dict:
    r = subprocess.run(["git", "show", f"{SINGLE_REF}:experiments/gpu-sim-u2o.7.1/measured.jsonl"],
                       cwd=REPO, capture_output=True, text=True, check=True)
    return {json.loads(l)["kernel_hash"]: json.loads(l)
            for l in r.stdout.splitlines() if l.strip()}


def run_mode(mode: str) -> dict:
    out = f"rows_{mode}.jsonl"
    cmd = [sys.executable, str(HERE / "batch_measure_worker.py"),
           "--mode", mode, "--out", out]
    r = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True, timeout=1200)
    if r.returncode != 0:
        raise SystemExit(f"FATAL: mode {mode} failed\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    rows = [json.loads(l) for l in (HERE / out).read_text().splitlines() if l.strip()]
    # the worker writes a FIXED summary filename, so read it immediately after
    # each run rather than assuming a per-mode file exists
    sp = HERE / "batch_summary.json"
    summ = json.loads(sp.read_text()) if sp.exists() else {}
    return {"rows": rows, "stdout": r.stdout, "summary": summ}


def compare(rows: list, single: dict) -> dict:
    comps = []
    for r in rows:
        s = single.get(r["kernel_hash"])
        if not s:
            continue
        a, b = r["median_runtime_ms"], s["median_runtime_ms"]
        rel = abs(a - b) / b
        tol = max(TOL_FLOOR, min(TOL_SIGMA * (s.get("ai_cv") or 0.0), TOL_CAP))
        comps.append({"rel": rel, "tol": tol, "ok": rel <= tol})
    n_ok = sum(1 for c in comps if c["ok"])
    rels = sorted(c["rel"] for c in comps)
    return {
        "rows_compared": len(comps),
        "rows_reproducing": n_ok,
        "fraction": n_ok / len(comps) if comps else 0.0,
        "median_rel_diff": statistics.median(rels) if rels else None,
        "p90_rel_diff": rels[int(len(rels) * 0.9)] if rels else None,
    }


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=2,
                    help="runs PER MODE. One run per mode is not enough: the "
                         "reproduction metric has run-to-run variance larger "
                         "than the difference between the modes, which is the "
                         "finding this script exists to establish.")
    args = ap.parse_args()
    single = single_run()
    results = {}
    for mode in MODES:
        reps = []
        for i in range(args.repeats):
            print(f"--- mode={mode} repeat {i+1}/{args.repeats} "
                  f"(protocol identical, isolation differs) ---", flush=True)
            r = run_mode(mode)
            reps.append(compare(r["rows"], single))
            print(f"    {reps[-1]['rows_reproducing']}/{reps[-1]['rows_compared']} "
                  f"reproduce, median rel diff "
                  f"{reps[-1]['median_rel_diff']*100:.1f}%", flush=True)
        fr = [x["fraction"] for x in reps]
        md = [x["median_rel_diff"] for x in reps]
        results[mode] = {
            "repeats": reps,
            "n_repeats": len(reps),
            "fraction_mean": sum(fr) / len(fr),
            "fraction_min": min(fr), "fraction_max": max(fr),
            "fraction_spread": max(fr) - min(fr),
            "median_rel_diff_mean": sum(md) / len(md),
            "rows_reproducing": [x["rows_reproducing"] for x in reps],
            "tests": {
                "warm": "baseline: state carries between examples",
                "cold": "removing carryover between EXAMPLES (tests H2)",
                "reset": "removing the shared context between INVOCATIONS (tests H1)",
            }[mode],
        }

    # ATTRIBUTION ONLY IF THE EFFECT EXCEEDS THE NOISE. The first version of
    # this picked the argmax fraction, and with one run per mode that selected
    # NOISE: warm scored 207, 189 and 173 rows reproducing across three runs of
    # the SAME configuration, which is a wider spread than any difference
    # between the modes. An argmax over noisy replicates is how a confident
    # wrong cause gets reported, which is the failure this whole experiment
    # exists to avoid.
    spread = max(r["fraction_spread"] for r in results.values())
    gaps = {m: results[m]["fraction_mean"] - results["warm"]["fraction_mean"]
            for m in MODES}
    best = max(MODES, key=lambda m: results[m]["fraction_mean"])
    gap = gaps[best]
    powered = gap > spread
    print(f"\nwithin-mode spread of the metric: {spread:.3f}")
    print(f"best mode {best!r} beats warm by {gap:+.3f}")
    print(f"ATTRIBUTION SUPPORTED: {powered}")
    causes = {
        "warm": "baseline that failed",
        "cold": ("H2 -- WARM STATE BETWEEN EXAMPLES. Removing the carryover is "
                 "what restores 7.1's medians, so the batching itself was fine "
                 "and the isolation is what a fix must restore."),
        "reset": ("H1 -- SHARED CUDA CONTEXT BETWEEN INVOCATIONS. 7.1's two "
                  "invocations were separate processes, so its ai_cv measured "
                  "context-to-context variation; sharing one context changes "
                  "what that statistic means."),
    }
    out = {
        "bead": "gpu-sim-u2o.6.7.1",
        "question": "which of the two candidate causes moved the medians?",
        "method": ("three runs, protocol IDENTICAL (40 timed launches, 2 "
                   "invocations x 20), varying only the ISOLATION. One thing at "
                   "a time, which is the discipline every wrong cause tonight "
                   "was missing."),
        "results": results,
        "best_mode": best,
        "gap_to_warm": gap,
        "within_mode_spread": spread,
        "attribution_supported": powered,
        "verdict": (causes[best] + f" (gap {gap:+.3f} against a within-mode "
                                  f"spread of {spread:.3f}, so the effect is "
                                  f"{'LARGER' if powered else 'NOT LARGER'} "
                                  f"than the noise)")
                    if powered else
                    "UNDERPOWERED -- NO ATTRIBUTION. The best mode beats warm by "
                    f"{gap:+.3f} while the SAME configuration varies by "
                    f"{spread:.3f} across repeats, so the modes are "
                    "indistinguishable at this sample size. Reporting H1 or H2 "
                    "here would be reporting noise with a mechanism attached, "
                    "which is worse than reporting nothing. Three runs of one "
                    "configuration gave 207, 189 and 173 rows reproducing.",
        "if_neither": ("the remaining suspect is the ESTIMATOR's context: 7.1 "
                       "measured each invocation in a process that had just "
                       "started, so its 20 trials include the driver's own "
                       "warm-up path. A long-lived process may simply report a "
                       "cleaner kernel time, in which case the batching is not "
                       "WRONG, it is measuring something slightly different, and "
                       "the fix is to state which rather than to pretend the two "
                       "are the same quantity."),
    }
    (HERE / "discriminator.json").write_text(json.dumps(out, indent=2))
    print("\n" + "=" * 68)
    for m in MODES:
        r = results[m]
        print(f"  {m:6s} rows reproducing {r['rows_reproducing']}  "
              f"mean {r['fraction_mean']*100:5.1f}%  "
              f"median-diff {r['median_rel_diff_mean']*100:5.1f}%  {r['tests']}")
    print(f"\nVERDICT: {out['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
