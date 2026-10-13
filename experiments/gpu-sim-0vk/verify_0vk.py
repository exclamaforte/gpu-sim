#!/usr/bin/env python3
"""gpu-sim-0vk VERIFY (CPU): acceptance items 1-9, loudly. Writes metrics.json.

Every number is recomputed here from committed artifacts; run.sh replays
merge + census + name_oob + this file and fails on any mismatch.
"""
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

FAIL = []
METRICS = {"bead": "gpu-sim-0vk", "stage": "verify"}


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name
          + (f" -- {detail}" if detail and not cond else ""))
    if not cond:
        FAIL.append(name)


def main() -> int:
    pred = json.loads((HERE / "prediction.json").read_text())
    keep = json.loads((HERE / "keep_list.json").read_text())
    man = json.loads((HERE / "manifest.json").read_text())
    rows = [json.loads(l)
            for l in (HERE / "measured_new.jsonl").read_text().splitlines()]
    census = json.loads((HERE / "census.json").read_text())
    guard = json.loads((HERE / "guard_verdicts.json").read_text())
    oob = json.loads((HERE / "oob_frozen.json").read_text())
    short = json.loads((HERE / "shortfall.json").read_text())

    # 1. predicted BEFORE generation, gate on record, manifest == keep-list
    check("item1: prediction gate passed pre-registered bars",
          pred["pass"] is True and pred["predicted_mm"] == 399
          and pred["predicted_measurable"] == 963,
          json.dumps(pred))
    check("item1: keep-list 990 cells", len(keep) == 990, str(len(keep)))
    check("item1: manifest 990 specs, 0 failures",
          man["n_specs"] == 990 and man["n_failures"] == 0,
          f"{man['n_specs']} specs {man['n_failures']} failures")
    keep_hashes = {c["kernel_hash"] for c in keep}
    check("item1: manifest hashes == keep-list hashes",
          {d["kernel_hash"] for d in man["specs"]} == keep_hashes)
    mm_pred = sum(1 for c in keep if c["predicted_class"] == "memory->memory")
    check("item1: predicted mm=399 (one short of ~400, named not rounded)",
          mm_pred == 399, str(mm_pred))
    METRICS["predicted"] = {"n": 990, "mm": mm_pred,
                            "meas": pred["predicted_measurable"]}

    # 2. measurable share of the NEW total >= 35%
    frozen_n = 2278
    meas_new = sum(1 for r in rows if r["measured_regime"] == "measurable")
    share = (352 + meas_new) / (frozen_n + len(rows))
    check("item2: measurable share >= 35%",
          share >= 0.35, f"{share:.3f}")
    check("item2: 983 new rows, 933 measurable",
          len(rows) == 983 and meas_new == 933,
          f"{len(rows)} rows {meas_new} measurable")
    METRICS["measurable_share"] = round(share, 4)

    # 3. memory->memory class >= ~400 (full corpus)
    full_mm = census["full_corpus_census"]["memory->memory"]
    check("item3: full-corpus memory->memory >= 400",
          full_mm >= 400, str(full_mm))
    check("item3: new-row mm=395 (shortfall took 4 predicted-mm cells)",
          census["mm_new"] == 395, str(census["mm_new"]))
    METRICS["mm_full"] = full_mm
    METRICS["mm_new"] = census["mm_new"]

    # 4. structurally varied families
    fam = Counter(r["op_type"] for r in rows)
    elem = {"add", "copy", "div", "mul", "relu", "saxpy", "scale", "triad"}
    elem_share = sum(n for o, n in fam.items() if o in elem) / len(rows)
    check("item4: >= 30 distinct families", len(fam) >= 30, str(len(fam)))
    check("item4: plain elementwise share < 10% (not same-ops-enlarged)",
          elem_share < 0.10, f"{elem_share:.3f}")
    check("item4: largest family share < 10%",
          max(fam.values()) / len(rows) < 0.10,
          f"{max(fam.values()) / len(rows):.3f}")
    METRICS["families"] = len(fam)
    METRICS["elem_share"] = round(elem_share, 4)

    # 5. frozen 2278 untouched by hash comparison; 4 known-OOB named
    counts = {}
    for p in ("experiments/gpu-sim-u2o.7.1/measured.jsonl",
              "experiments/gpu-sim-e5q/measured.jsonl",
              "experiments/gpu-sim-u2o.11.3/measured.jsonl",
              "experiments/gpu-sim-f6y/measured_new.jsonl"):
        counts[p] = sum(1 for _ in (REPO / p).read_text().splitlines())
    check("item5: frozen counts 264/192/1804/18",
          counts == {"experiments/gpu-sim-u2o.7.1/measured.jsonl": 264,
                     "experiments/gpu-sim-e5q/measured.jsonl": 192,
                     "experiments/gpu-sim-u2o.11.3/measured.jsonl": 1804,
                     "experiments/gpu-sim-f6y/measured_new.jsonl": 18},
          json.dumps(counts))
    snap = (REPO / "experiments/gpu-sim-f6y/u113_measured_sha256_before.txt"
            ).read_text().split()[0]
    now = hashlib.sha256(
        (REPO / "experiments/gpu-sim-u2o.11.3/measured.jsonl"
         ).read_bytes()).hexdigest()
    check("item5: 11.3 sha256 == f6y pre-measurement snapshot",
          now == snap, f"{now[:16]} vs {snap[:16]}")
    frozen_hashes = set()
    for p in counts:
        for line in (REPO / p).read_text().splitlines():
            frozen_hashes.add(json.loads(line)["kernel_hash"])
    check("item5: frozen hashes 2278 unique",
          len(frozen_hashes) == 2278, str(len(frozen_hashes)))
    check("item5: new rows disjoint from frozen (append-only)",
          not (frozen_hashes & {r["kernel_hash"] for r in rows}))
    check("item5: 4 known-OOB rows named and all frozen, none re-measured",
          oob["n"] == 4 and {r["kernel_hash"] for r in oob["rows"]}
          <= frozen_hashes and not ({r["kernel_hash"] for r in oob["rows"]}
                                    & {r["kernel_hash"] for r in rows}),
          json.dumps([r["kernel_hash"] for r in oob["rows"]]))
    METRICS["frozen_sha256_113"] = now[:16]

    # 6. every new template passed the OOB guard
    check("item6: 38/38 templates guard-pass, 0 refused",
          guard["n_templates"] == 38 and not guard["refused"],
          json.dumps(guard)[:200])
    check("item6: every manifest op has a guard verdict",
          {d["op_type"] for d in man["specs"]} <= set(
              guard["per_template"]))
    check("item6: every row carries guard_passed_template",
          all(r.get("guard_passed_template") is True for r in rows))

    # 7. per-cell provenance
    check("item7: ai_cv on all 983 rows",
          all(r.get("ai_cv") is not None for r in rows))
    check("item7: own_job_id on all rows",
          all(r.get("queue_state", {}).get("own_job_id") for r in rows))
    check("item7: queue_state + protocol on all rows",
          all(r.get("queue_state", {}).get("n_other_active") is not None
              and r.get("protocol") == "single_run_stages_2_5"
              for r in rows))

    # 8. compile unqueued / measure queued, durations recorded
    check("item8: merge shortfall 7 named (no silent drops)",
          len(short) == 7 and len(rows) + len(short) == 990,
          f"{len(short)} shortfall")
    n_lb = sum(1 for r in rows if r["measured_regime"] == "launch_bound")
    check("item8: 50 launch_bound + 933 measurable = 983",
          n_lb == 50 and n_lb + meas_new == 983, str(n_lb))
    METRICS["n_rows"] = len(rows)
    METRICS["n_shortfall"] = len(short)

    # 9. negative control demonstrated red
    neg = census["neg_control"]
    check("item9: neg control PASS (gate FAILed the 90%-one-regime mix)",
          neg["control"] == "PASS" and neg["gate_verdict"] == "FAIL",
          json.dumps(neg))
    check("item9: census CENSUS PASS (bar + full gate + control)",
          census["bead_bar"] == "PASS" and census["full_gate"] == "PASS",
          f"{census['bead_bar']}/{census['full_gate']}")

    (HERE / "metrics.json").write_text(json.dumps(METRICS, indent=2))
    print(f"\nVERIFY {'PASS' if not FAIL else 'FAIL: ' + str(FAIL)}")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    raise SystemExit(main())
