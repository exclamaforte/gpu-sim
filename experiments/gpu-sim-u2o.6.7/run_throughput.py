#!/usr/bin/env python3
"""Real throughput for the measurement stage. gpu-sim-u2o.6.7.

Three things, in order of importance:

  1. The REAL number, from 7.1's committed batch (ts 20260928t074004-b429f469),
     derived TWO independent ways and cross-checked.
  2. The bottleneck answer the bead was opened for: what is the wall clock
     actually spent on.
  3. The fabricated artifact this bead was reopened for, run through the new
     guard to show it is now REFUSED rather than quotable.

No GPU and no ts job: every number comes from a committed artifact. The point
of the bead is that measurement throughput should be tracked as a real metric,
not that it needs re-measuring.
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from epic_u2o.throughput_evidence import (  # noqa: E402
    ThroughputEvidenceError, bottleneck_split, cross_check,
    throughput_from_timestamps, throughput_from_wall,
)

REF = "lab/gpu-sim-u2o.7.1"
ROWS = "experiments/gpu-sim-u2o.7.1/measured.jsonl"
SUMMARY = "experiments/gpu-sim-u2o.7.1/measure_summary.json"
FABRICATED = REPO / "runs/logs/throughput_demo_run.jsonl"


def git_show(path: str) -> str:
    r = subprocess.run(["git", "show", f"{REF}:{path}"], cwd=REPO,
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(
            f"FATAL: cannot read {REF}:{path}\n{r.stderr}\n"
            f"This experiment reports REAL throughput. A synthetic stand-in "
            f"would be the exact failure this bead exists to correct.")
    return r.stdout


def main() -> int:
    rows = [json.loads(l) for l in git_show(ROWS).splitlines() if l.strip()]
    summary = json.loads(git_show(SUMMARY))
    wall = summary["wall_sec"]
    launches = rows[0]["n_trials"] * rows[0]["n_invocations"]

    a = throughput_from_wall(rows, wall, "measure_summary.wall_sec (harness clock)")
    b = throughput_from_timestamps(rows)
    check = cross_check(a, b)
    split = bottleneck_split(rows, wall, launches)

    out = {
        "bead": "gpu-sim-u2o.6.7",
        "source": f"{REF}:{ROWS} and {REF}:{SUMMARY}",
        "ts_job": "20260928t074004-b429f469",
        "gpu_used": False,
        "n_rows_measured": summary["n_measured"],
        "n_failed": summary["n_failed"],
        "throughput": {
            "harness_wall": a.to_dict(),
            "timestamp_span": b.to_dict(),
            "cross_check": check,
        },
        "bottleneck": split,
    }

    # --- the fabricated artifact, now refused ----------------------------
    fab_report: dict = {"path": str(FABRICATED.relative_to(REPO))}
    if FABRICATED.exists():
        fab_rows = [json.loads(l) for l in
                    FABRICATED.read_text().splitlines() if l.strip()]
        try:
            throughput_from_wall(fab_rows, 0.15, "fabricated file")
            fab_report["refused"] = False
            fab_report["problem"] = "IT WAS ACCEPTED -- the guard is broken"
        except ThroughputEvidenceError as exc:
            fab_report["refused"] = True
            fab_report["reason"] = str(exc)
        # and what it claimed, for the record
        claimed = fab_rows[0].get("throughput_examples_per_sec")
        if claimed:
            fab_report["claimed_examples_per_sec"] = claimed
            fab_report["claimed_examples_per_hour"] = claimed * 3600
            fab_report["overstatement_vs_real"] = claimed / a.examples_per_sec
        # the rows with nothing in them
        empty = [r for r in fab_rows if not r.get("trials")]
        fab_report["n_rows_with_no_trials_but_marked_completed"] = len(empty)
        fab_report["n_distinct_kernel_hash"] = len({r["kernel_hash"] for r in fab_rows})
        fab_report["was_tracked_in_git"] = subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(FABRICATED.relative_to(REPO))],
            cwd=REPO, capture_output=True).returncode == 0
    else:
        # runs/ is UNTRACKED, so it exists only in the main checkout and not in
        # a worktree. Saying "absent" without looking in the main checkout would
        # be the convenient reading and the wrong one, so look there and say
        # plainly which tree it was found in.
        # REPO is .lab/wt/<bead>, so the MAIN checkout is three levels up.
        # Guarded by a marker so a wrong guess reports "not found" rather than
        # reading an unrelated directory.
        main_checkout = REPO.parents[2] if len(REPO.parents) > 2 else REPO
        if not (main_checkout / ".lab").is_dir():
            main_checkout = REPO
        alt = main_checkout / "runs/logs/throughput_demo_run.jsonl"
        fab_report["not_in_this_worktree"] = True
        fab_report["searched_main_checkout"] = str(alt)
        if alt.exists():
            fab_rows = [json.loads(l) for l in
                        alt.read_text().splitlines() if l.strip()]
            try:
                throughput_from_wall(fab_rows, 0.15, "fabricated file")
                fab_report["refused"] = False
                fab_report["problem"] = "IT WAS ACCEPTED -- the guard is broken"
            except ThroughputEvidenceError as exc:
                fab_report["refused"] = True
                fab_report["reason"] = str(exc)
            claimed = fab_rows[0].get("throughput_examples_per_sec", 0)
            fab_report["n_rows"] = len(fab_rows)
            fab_report["n_distinct_kernel_hash"] = len(
                {r["kernel_hash"] for r in fab_rows})
            fab_report["n_rows_with_no_trials_but_marked_completed"] = sum(
                1 for r in fab_rows if not r.get("trials"))
            if claimed:
                fab_report["claimed_examples_per_sec"] = claimed
                fab_report["claimed_examples_per_hour"] = claimed * 3600
                fab_report["overstatement_vs_real"] = claimed / a.examples_per_sec
            fab_report["was_tracked_in_git"] = subprocess.run(
                ["git", "ls-files", "--error-unmatch",
                 "runs/logs/throughput_demo_run.jsonl"],
                cwd=REPO, capture_output=True).returncode == 0
        else:
            fab_report["refused"] = None
            fab_report["problem"] = "not present in the worktree or main checkout"
    out["fabricated_artifact"] = fab_report

    (HERE / "metrics.json").write_text(json.dumps(out, indent=2))
    # a REAL log in the schema the tracker already writes, so it is consumable
    (HERE / "throughput_real_run.jsonl").write_text("".join(
        json.dumps({
            "run_id": "7_1_b429f469",
            "phase": "measurement",
            "kernel_hash": r["kernel_hash"],
            "status": "completed",
            "duration_sec": None,   # not per-row recoverable; wall is aggregate
            "trials": r["invocation_medians_ms"],
            "median_ms": r["median_runtime_ms"],
            "measured_at": r["measured_at"],
            "n_trials": r["n_trials"],
            "n_invocations": r["n_invocations"],
            "throughput_examples_per_sec": a.examples_per_sec,
            "metadata": {"problem_size": r["problem_size"],
                         "op_type": r["op_type"],
                         "measured_regime": r["measured_regime"],
                         "ts_job": "20260928t074004-b429f469"},
        }) + "\n" for r in rows))

    print(f"REAL throughput from {out['n_rows_measured']} measured rows "
          f"(ts {out['ts_job']}):")
    print(f"  harness wall clock : {a.examples_per_sec:.4f} ex/s = "
          f"{a.examples_per_hour:,.0f} ex/hour")
    print(f"  timestamp span     : {b.examples_per_sec:.4f} ex/s = "
          f"{b.examples_per_hour:,.0f} ex/hour")
    print(f"  cross-check        : {check['disagreement_pct']:.2f}% disagreement "
          f"({'AGREE' if check['agree_within_tolerance'] else 'DISAGREE'})")
    print(f"\nbottleneck: {split['verdict']}")
    print(f"  {split['overhead_fraction_of_wall']*100:.2f}% overhead, "
          f"{split['kernel_fraction_of_wall']*100:.3f}% kernel")
    print(f"  {split['wall_sec_per_launch']*1e3:.2f} ms wall per launch vs "
          f"{split['kernel_us_per_launch_median']:.1f} us of kernel")
    if fab_report.get("refused"):
        print(f"\nfabricated artifact: REFUSED -- {fab_report['reason'][:110]}")
        print(f"  claimed {fab_report['claimed_examples_per_hour']:,.0f} ex/hour, "
              f"{fab_report['overstatement_vs_real']:.1f}x the real rate")
    return 0 if check["agree_within_tolerance"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
