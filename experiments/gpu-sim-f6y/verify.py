#!/usr/bin/env python3
"""gpu-sim-f6y VERIFY (CPU): acceptance items, loudly.

Item 3: every shipped row has >= 2 successful invocations (ai_cv needs 2);
  zero invocations = no row, enforced by the measure step and re-checked here.
Item 4: the 18 rows' regimes are reported (census.json), not assumed.
Item 6: the negative control lives in tests/test_oob_guard.py (pinned broken
  sources refused with exact maxima); this verifies the fixed cells passed
  the same guard at build (manifest failures == 0).
Item 7: the 11.3 corpus is FROZEN -- measured.jsonl sha256 must equal the
  pre-measurement snapshot, and the 11.3 directory must be git-clean.
Item 8: census.json sums 1474 + 237 + 111 = 1822 with verdict PASS.
"""
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
U113 = REPO / "experiments" / "gpu-sim-u2o.11.3"

FAIL = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name
          + (f" ({detail})" if detail and not cond else ""))
    if not cond:
        FAIL.append(name)


def main() -> int:
    man = json.loads((HERE / "manifest.json").read_text())
    check("manifest 80 specs, 0 build failures",
          man["n_specs"] == 80 and man["n_failures"] == 0,
          json.dumps(man.get("failures", []))[:200])
    rows = [json.loads(l) for l in
            (HERE / "measured_new.jsonl").read_text().splitlines()]
    check("18 rows shipped", len(rows) == 18, f"got {len(rows)}")
    ops = Counter(r["op_type"] for r in rows)
    check("11 seg + 7 strided", ops == {"segmented_read": 11,
                                        "strided_write": 7}, str(dict(ops)))
    check("all stage3", {r["stage"] for r in rows} == {"stage3"})
    check("ai_cv on all 18 (item 3: >=2 invocations each)",
          all(r.get("ai_cv") is not None and r.get("n_invocations", 0) >= 2
              for r in rows))
    check("protocol stamped (item 3: no row without it)",
          {r.get("protocol") for r in rows} == {"single_run_stages_2_5"})
    check("driver_scalars + manifest_occurrences joined",
          all(r.get("driver_scalars") and r.get("manifest_occurrences")
              for r in rows))
    qs = [r.get("queue_state", {}) for r in rows]
    check("queue_state with own_job_id on all 18",
          all(q.get("own_job_id") for q in qs))
    job = json.loads((HERE / "job.json").read_text())
    check("own_job_id is the f6y job",
          {q.get("own_job_id") for q in qs} == {job["queue_jobs"]["measure"]},
          str({q.get("own_job_id") for q in qs}))
    old_rows = [json.loads(l) for l in
                (U113 / "measured.jsonl").read_text().splitlines()]
    old_hashes = {r["kernel_hash"] for r in old_rows}
    new_hashes = {r["kernel_hash"] for r in rows}
    check("18 new hashes, none in the 1804",
          len(new_hashes) == 18 and not (new_hashes & old_hashes))
    missing = set(json.loads(
        (U113 / "measure_summary.json").read_text())["missing_hashes"])
    check("shortfall hashes stay missing (broken-template record kept)",
          len(missing) == 18 and not (new_hashes & missing))
    s = json.loads((HERE / "measure_summary.json").read_text())
    check("contention disposition recorded",
          "contention_disposition" in s,
          "no contention block in measure_summary.json")
    comp = json.loads((HERE / "build" / "compile_summary.json").read_text())
    check("compiled 80/80 unqueued",
          comp["n_ok"] == 80 and comp["n_failed"] == 0,
          json.dumps(comp))
    sha = hashlib.sha256((U113 / "measured.jsonl").read_bytes()).hexdigest()
    snap = (HERE / "u113_measured_sha256_before.txt").read_text().split()[0]
    check("11.3 measured.jsonl byte-identical (item 7)",
          sha == snap, f"{sha[:12]} != {snap[:12]}")
    p = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain",
                        "--", "experiments/gpu-sim-u2o.11.3/"],
                       capture_output=True, text=True)
    check("11.3 directory git-clean (item 7)", p.stdout.strip() == "",
          p.stdout[:200])
    cen = json.loads((HERE / "census.json").read_text())
    check("census 1474/237/111 = 1822 PASS (item 8)",
          cen["corpus_total"] == 1822 and cen["verdict"] == "PASS"
          and cen["corpus_census"] == {"latency->latency": 1474,
                                       "latency->memory": 237,
                                       "memory->memory": 111},
          json.dumps(cen.get("corpus_census")))
    print(f"\nVERIFY {'PASS' if not FAIL else 'FAIL: ' + str(FAIL)}")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    raise SystemExit(main())
