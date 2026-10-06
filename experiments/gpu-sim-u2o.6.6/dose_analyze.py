#!/usr/bin/env python3
"""gpu-sim-u2o.6.6 dose analysis (CPU): apply ruling-(b) to the gapless dose,
prove the queue '1' is self-misattribution, compare regimes dose-vs-anchor.
Deterministic; writes metrics JSON to --metrics-out.

Inputs: experiments/gpu-sim-u2o.6.6/dose/family_{launch,memory,gemm,reduce}.json
(copies of the ts-job outputs, sha256-verified against sha256sums.txt) plus
the committed anchor_v1 family files.
"""
import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DOSE = HERE / "dose"
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from epic_u2o import measurement_stats as ms  # noqa: E402

DOSE_JOB = "20261006t041800-5878994f"


def load(name, base):
    return json.loads((base / name).read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-out", required=True)
    args = ap.parse_args()
    fails = []

    # Integrity: dose copies match the job's sha256sums.
    sums = {}
    for line in (DOSE / "sha256sums.txt").read_text().splitlines():
        h, p = line.split()
        sums[Path(p).name] = h
    for f in ["family_launch.json", "family_memory.json",
              "family_gemm.json", "family_reduce.json"]:
        h = hashlib.sha256((DOSE / f).read_bytes()).hexdigest()
        if sums.get(f) != h:
            fails.append(f"{f} sha256 mismatch: job output altered in copy")

    # Ruling-(b) dispositions on gapless signals + self-attribution proof.
    fams, cells = {}, []
    for fam in ("launch", "memory", "gemm", "reduce"):
        d = load(f"family_{fam}.json", DOSE)
        disp = ms.apply_contention_rule(d["telemetry_before"]["fields"],
                                        d["telemetry_after"]["fields"],
                                        d["queue_state_before"],
                                        d["queue_state_after"])
        sides = {}
        for side in ("queue_state_before", "queue_state_after"):
            q = d[side]
            oids = [j["id"] for j in q.get("other_active_jobs", [])]
            rids = [j["id"] for j in q.get("other_running_jobs", [])]
            sides[side] = {
                "raw_n_other_active": q.get("n_other_active"),
                "raw_other_ids_active": oids,
                "raw_other_ids_running": rids,
                "running_self_only": sorted(rids) in ([], [DOSE_JOB]),
                "stale_own_job_id_recorded": q.get("own_job_id"),
                "true_own_job_id": DOSE_JOB,
            }
        external = sorted({i for s in sides.values() for i in s["raw_other_ids_running"]} - {DOSE_JOB})
        attributed_keep = all(s["running_self_only"] for s in sides.values())
        disp["self_attribution"] = {
            "per_boundary": sides,
            "external_tenant_ids": external,
            "attributed_disposition": "keep" if attributed_keep and not external else disp["disposition"],
            "attribution_basis": "PI ruling 2026-10-06 option (b): entries id-equal to the dose job attributed out; any other id still discards",
            "reading": "queue '1' is the dose job itself on both boundaries (id match), no external tenant; recorded own_job_id is stale (matcher predates dose.sh), so contention-freedom rests on gapless telemetry",
        }
        fams[fam] = disp
        final = disp["self_attribution"]["attributed_disposition"]
        if final == "discard":
            fails.append(f"dose family {fam} trips ruling-b: {disp['fired_clauses']}")
        for c in d["results"]:
            cells.append((fam, c))

    # Regime sigmas, dose population.
    regime = {}
    for r in ("sub-10us", "over-10us"):
        xs = [c["across_invocation_cv"] for _, c in cells
              if ms.regime_of(c["median_ms"]) == r]
        regime[r] = {"n": len(xs),
                     "median_cv": round(statistics.median(xs), 4)}

    # Dose-vs-anchor per-cell drift (same join key as the variance probe).
    anchor = {}
    for fam in ("launch", "memory", "gemm", "reduce"):
        for c in load(f"family_{fam}.json",
                      REPO / "experiments" / "gpu-sim-u2o.6.6.2")["results"]:
            anchor[c["label"]] = c["median_ms"]
    ratios = []
    for _, c in cells:
        a = anchor.get(c["label"])
        if a:
            ratios.append(round(c["median_ms"] / a, 4))
    ratios.sort()
    q = lambda p: ratios[min(int(p * len(ratios)), len(ratios) - 1)]
    drift = {"n": len(ratios), "min": ratios[0], "p10": q(0.10),
             "median": q(0.50), "p90": q(0.90), "max": ratios[-1]}

    # Carryover trajectory (raw boundary temps from the dose files).
    traj = []
    for f in ("launch", "memory", "gemm", "reduce"):
        _d = load(f"family_{f}.json", DOSE)
        _tb = _d["telemetry_before"]["fields"]
        _ta = _d["telemetry_after"]["fields"]
        traj.append({"family": f, "before_C": float(_tb["temperature.gpu"].split()[0]), "after_C": float(_ta["temperature.gpu"].split()[0]), "wall_s": round(_d["wall_sec"], 1)})

    metrics = {
        "bead": "gpu-sim-u2o.6.6",
        "dose": {"ts_job": DOSE_JOB, "queue": "empty at submit (verified), "
                 "one batched job, 4 families sequential",
                 "n_cells": len(cells)},
        "ruling_b_dispositions": fams,
        "regime_cv_dose": regime,
        "dose_vs_anchor_v1": {
            "drift_dose_over_anchor": drift,
            "reading": "second run-to-run instance beside variance_probe "
                       "(v8/v6 median 1.0181): independent corroboration of "
                       "the 6.6 spread discipline",
        },
        "carryover_trajectory": traj,
        "followup_wrapper_fix": "dose.sh wraps run_family.sh, so the queue matcher (own_command_substr='run_family.sh <fam>') resolves a STALE job id and self leaks into n_other_*. Next dose: match own job by id/description (e.g. U2O_JOB_DESC='u2o.6.6 dose' + description-substring match), so own_job_entry resolves and the queue axis is verifiable in-record. Named per PI ruling 2026-10-06; not implemented here (no unverified harness change).",
        "verdict": "pass" if not fails else "fail",
        "failures": fails,
    }
    Path(args.metrics_out).write_text(json.dumps(metrics, indent=2,
                                                 sort_keys=True) + "\n")
    print(f"cells={len(cells)} drift_median={drift['median']} "
          f"p90={drift['p90']} verdict={metrics['verdict']} -> {args.metrics_out}",
          flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
