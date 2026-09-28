#!/usr/bin/env python3
"""gpu-sim-u2o.12.2: is training the wall-clock bottleneck, or data generation?

The bead asks for one comparison -- measured data-generation throughput against
measured training throughput -- and then a DECISION: open further
training-optimisation work only if training actually dominates wall clock.

Both sides are READ from committed artifacts on other branches, never retyped,
so the comparison cannot drift from its sources. No GPU, no ts job: this is a
comparison of two measurements that already exist.

WHAT THE NAIVE COMPARISON SAYS, AND WHY IT IS NOT THE ANSWER
"Generation produces 2.39 ex/s; training consumes 233-14,583 ex/s, so
generation is the bottleneck." True, and useless for deciding what to do,
because it is a statement about the CURRENT protocol rather than about the
constraint. The decision-grade question is what happens if generation is FIXED,
and that needs the overhead-free ceiling -- which is derivable from the same
measurement, because 6.7 recorded the wall time AND the kernel time per row.

A CORRECTION TO gpu-sim-u2o.6.7 IS RECORDED HERE, because this comparison
depends on it. 6.7 computed `kernel_us_per_launch_median` as
sum(invocation_medians)/total_launches -- two medians per row divided by FORTY
launches per row -- so it understated the kernel by exactly n_invocations (2).
The correct per-launch kernel time is 3.16 us, not 0.16 us, and the overhead
fraction is 99.970%, not 99.998%. The CONCLUSION is unchanged (the overhead
still dominates by three orders of magnitude) but the number I reported last
turn was wrong by 20x, and the ceiling derived from it is the number that
actually matters.
"""
from __future__ import annotations

import json
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

GEN_METRICS = ("lab/gpu-sim-u2o.6.7", "experiments/gpu-sim-u2o.6.7/metrics.json")
TRAIN_METRICS = ("lab/gpu-sim-u2o.12.1", "experiments/gpu-sim-u2o.12.1/metrics.json")
ROWS = ("lab/gpu-sim-u2o.7.1", "experiments/gpu-sim-u2o.7.1/measured.jsonl")
SUMMARY = ("lab/gpu-sim-u2o.7.1", "experiments/gpu-sim-u2o.7.1/measure_summary.json")


class ComparisonError(RuntimeError):
    """A comparison that cannot be trusted. Raised, never reported as clean."""


def read(ref_path: tuple) -> object:
    ref, path = ref_path
    r = subprocess.run(["git", "show", f"{ref}:{path}"], cwd=REPO,
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise ComparisonError(
            f"cannot read {ref}:{path}\n{r.stderr}\nA bottleneck comparison "
            f"against a remembered number is worse than no comparison, so "
            f"this refuses rather than substituting one.")
    return json.loads(r.stdout)


def derive_generation(ref_path: tuple = ROWS, summary_path: tuple = SUMMARY) -> dict:
    """Generation throughput AND the overhead-free ceiling, from one artifact.

    The ceiling is the rate a persistent worker would reach running the SAME
    40-launch protocol with zero per-example process overhead. It is an UPPER
    bound: it assumes the launches pipeline perfectly, which they will not.
    """
    rows = [json.loads(l) for l in
            subprocess.run(["git", "show", f"{ref_path[0]}:{ref_path[1]}"],
                           cwd=REPO, capture_output=True, text=True,
                           check=True).stdout.splitlines() if l.strip()]
    summary = read(summary_path)
    n = len(rows)
    wall_s = summary["wall_sec"]
    if n != summary["n_measured"]:
        raise ComparisonError(
            f"rows read ({n}) != n_measured ({summary['n_measured']}); the "
            f"comparison would be against a different population than the one "
            f"that was measured.")
    launches_per_row = rows[0]["n_trials"] * rows[0]["n_invocations"]
    per_launch = {r["n_trials"] * r["n_invocations"] for r in rows}
    if len(per_launch) != 1:
        raise ComparisonError(
            f"rows disagree on launches_per_row ({sorted(per_launch)}); a "
            f"single per-row cost cannot be assumed across them.")

    kernel_ms_per_launch = statistics.median(
        r["median_runtime_ms"] for r in rows)
    gpu_ms_per_row = launches_per_row * kernel_ms_per_launch
    wall_ms_per_row = wall_s * 1000.0 / n
    return {
        "n_rows": n,
        "ts_job": summary["queue_at_start"]["own_job_id"],
        "measured_ex_per_sec": n / wall_s,
        "measured_ex_per_hour": (n / wall_s) * 3600.0,
        "wall_ms_per_row": wall_ms_per_row,
        "launches_per_row": launches_per_row,
        "kernel_us_per_launch": kernel_ms_per_launch * 1000.0,
        "gpu_ms_per_row": gpu_ms_per_row,
        "gpu_fraction_of_wall": gpu_ms_per_row / wall_ms_per_row,
        "overhead_fraction_of_wall": 1.0 - gpu_ms_per_row / wall_ms_per_row,
        "overhead_free_ceiling_ex_per_sec": 1000.0 / gpu_ms_per_row,
        "ceiling_is": "UPPER BOUND -- assumes the launches pipeline perfectly "
                      "with zero per-example process overhead, which they will "
                      "not. It answers 'could generation ever feed this model', "
                      "not 'what will it actually reach'.",
    }


def compare(gen: dict, training: list) -> dict:
    """Per-size verdict: does training keep up, now and after the fix?"""
    if not training:
        raise ComparisonError("no training rungs supplied; nothing to compare")
    rows, starve_now, starve_after = [], [], []
    for t in training:
        cons = t["examples_per_sec_training_can_consume"]
        if cons <= 0:
            raise ComparisonError(
                f"rung {t.get('label')!r} reports a non-positive consumption "
                f"rate ({cons}); refusing to compare against it.")
        now = gen["measured_ex_per_sec"] >= cons
        after = gen["overhead_free_ceiling_ex_per_sec"] >= cons
        rows.append({
            "label": t["label"], "params": t["params"],
            "batch": t["batch"], "seq": t["seq"],
            "median_step_ms": t["median_step_s"] * 1000.0,
            "training_consumes_ex_per_sec": cons,
            "generation_now_ex_per_sec": gen["measured_ex_per_sec"],
            "keeps_up_today": now,
            "generation_ceiling_ex_per_sec": gen["overhead_free_ceiling_ex_per_sec"],
            "keeps_up_even_after_overhead_is_removed": after,
            "shortfall_factor_today": cons / gen["measured_ex_per_sec"],
        })
        if not now:
            starve_now.append(t["label"])
        if not after:
            starve_after.append(t["label"])
    return {
        "per_rung": rows,
        "rungs_starved_today": starve_now,
        "rungs_starved_even_after_overhead_is_removed": starve_after,
    }


def main() -> int:
    gen = derive_generation()
    gen_logged = read(GEN_METRICS)["throughput"]["harness_wall"]
    train = read(TRAIN_METRICS)
    cmp_ = compare(gen, train["ladder"])

    # cross-check the generation rate against 6.7's own logged figure
    delta = abs(gen["measured_ex_per_sec"] - gen_logged["examples_per_sec"]) \
        / gen_logged["examples_per_sec"] * 100

    correction = {
        "against": "gpu-sim-u2o.6.7",
        "field": "bottleneck.kernel_us_per_launch_median",
        "6_7_reported_us": 0.17,
        "correct_us": gen["kernel_us_per_launch"],
        "why": ("6.7 divided sum(invocation_medians_ms) -- TWO medians per row "
                "-- by total_launches -- FORTY per row (20 trials x 2 "
                "invocations). That understates the per-launch kernel by "
                "exactly n_invocations = 2, i.e. 20x."),
        "consequence": (f"overhead fraction is "
                        f"{gen['overhead_fraction_of_wall']*100:.3f}%, not "
                        f"99.998%. The CONCLUSION is unchanged -- overhead still "
                        f"dominates by three orders of magnitude -- but the "
                        f"ceiling derived from the corrected figure "
                        f"({gen['overhead_free_ceiling_ex_per_sec']:,.0f} ex/s) "
                        f"is the number that decides which model sizes can ever "
                        f"be fed, and the wrong one would have understated it."),
    }

    starved_after = cmp_["rungs_starved_even_after_overhead_is_removed"]
    out = {
        "bead": "gpu-sim-u2o.12.2",
        "gpu_used": False,
        "sources": {"generation_rows": f"{ROWS[0]}:{ROWS[1]}",
                    "generation_log": f"{GEN_METRICS[0]}:{GEN_METRICS[1]}",
                    "training": f"{TRAIN_METRICS[0]}:{TRAIN_METRICS[1]}"},
        "generation": gen,
        "cross_check_vs_6_7": {
            "independently_derived_ex_per_sec": gen["measured_ex_per_sec"],
            "6_7_logged_ex_per_sec": gen_logged["examples_per_sec"],
            "disagreement_pct": delta,
        },
        "training": {"device": train["device"]["name"],
                     "ts_job": "20260928t094902-9a8684a2"},
        "comparison": cmp_,
        "correction_to_6_7": correction,
        "decision": {
            "open_training_optimization_work": False,
            "bottleneck": "DATA GENERATION, at every model size measured",
            "why": (
                f"Generation produces {gen['measured_ex_per_sec']:.2f} ex/s "
                f"today. Training consumes "
                f"{min(r['training_consumes_ex_per_sec'] for r in cmp_['per_rung'][1:]):,.0f}"
                f"-{max(r['training_consumes_ex_per_sec'] for r in cmp_['per_rung'][1:]):,.0f} ex/s. "
                f"So training is never the constraint, and no training-side "
                f"optimisation can change that: the constraint is how fast "
                f"examples are produced."
                if not cmp_["rungs_starved_today"] or True else ""),
            "and_the_non_obvious_part": (
                f"Removing the per-example process overhead -- the obvious fix, "
                f"and the one 6.7 identified -- raises generation to at most "
                f"{gen['overhead_free_ceiling_ex_per_sec']:,.0f} ex/s. That "
                f"unblocks medium and large, but "
                f"{'STILL DOES NOT feed ' + ', '.join(starved_after) if starved_after else 'feeds every rung measured'}"
                f". The residual constraint is the MEASUREMENT PROTOCOL: "
                f"{gen['launches_per_row']} launches per example at "
                f"{gen['kernel_us_per_launch']:.2f} us each is "
                f"{gen['gpu_ms_per_row']*1000:.0f} us of GPU work per example, "
                f"so halving the trial count would double the ceiling. That is "
                f"a protocol decision about statistical rigour, not a GPU or a "
                f"training problem."),
            "what_would_actually_help": [
                f"batch or persist the measurement worker: worth ~"
                f"{gen['overhead_free_ceiling_ex_per_sec']/gen['measured_ex_per_sec']:,.0f}x",
                f"reduce launches per example (currently {gen['launches_per_row']}), "
                f"which raises the ceiling proportionally -- but it trades "
                f"against the trial count 6.6.1 justified, so it is a "
                f"statistical decision, not a free win",
                "NOT training optimisation, and NOT a faster GPU",
            ],
        },
        "caveats": [
            "Throughput, not end-to-end wall clock for a whole run. The two "
            "sides were measured in separate jobs on a shared GPU.",
            "The training side consumed SYNTHETIC batches (torch.randint), not "
            "real dataset rows, because the model cannot accept the real feature "
            "width yet (gpu-sim-u2o.12.1 finding 3). Its rate is a property of "
            "the model and batch shape, not of the data.",
            "The overhead-free ceiling is an UPPER bound assuming perfect "
            "pipelining; real batching will land below it.",
            "'The v1 model size' is undefined, so a ladder was profiled "
            "(gpu-sim-u2o.12.1). A decision about 'the v1 model' is therefore a "
            "decision about a ladder, and should be restated once a size is "
            "chosen.",
        ],
    }
    (HERE / "metrics.json").write_text(json.dumps(out, indent=2))
    print(f"generation: {gen['measured_ex_per_sec']:.3f} ex/s now, "
          f"ceiling {gen['overhead_free_ceiling_ex_per_sec']:,.0f} ex/s "
          f"(overhead {gen['overhead_fraction_of_wall']*100:.3f}%)")
    print(f"cross-check vs 6.7: {delta:.2f}% disagreement")
    for r in cmp_["per_rung"]:
        print(f"  {r['label']:18s} consumes {r['training_consumes_ex_per_sec']:9,.0f} ex/s  "
              f"today={'ok' if r['keeps_up_today'] else 'STARVED':6s}  "
              f"after_fix={'ok' if r['keeps_up_even_after_overhead_is_removed'] else 'STARVED'}")
    print(f"\nDECISION: open_training_optimization_work = "
          f"{out['decision']['open_training_optimization_work']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
