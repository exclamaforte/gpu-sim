#!/usr/bin/env python3
"""gpu-sim-u2o.14 DOSE MEASURE (GPU, through ts): prebuilt dose binaries.

2 invocations x 20 trials per cell via SafeHarness; rows carry ai_cv,
own_job_id, queue_state, protocol dose_14. Prints GPQUEUE_PROGRESS.
Writes dose_measured.jsonl + dose_summary.json (CPU-side merge asserts
one row per manifest hash: the spec-id -> trial-row pairing).
"""
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from epic_u2o.safe_harness import SafeHarness  # noqa: E402
from epic_u2o import queue_state as qs  # noqa: E402
from epic_u2o import corpus_protocol as cp  # noqa: E402

TRIALS = 20
INVOCATIONS = 2
PROTOCOL = "dose_14"


def main() -> int:
    man = json.loads((HERE / "dose_manifest.json").read_text())
    build = HERE / "build"
    q0 = qs.queue_state(own_description_substr="gpu-sim-u2o.14",
                        own_command_substr="dose_measure_14")
    h = SafeHarness(timeout_sec=600)
    rows, failures = [], []
    t0 = time.time()
    with (HERE / "build" / "dose_raw.jsonl").open("w") as fh:
        for i, d in enumerate(man["specs"]):
            exe = build / f"k_{d['kernel_hash']}.out"
            if not exe.exists():
                failures.append({"hash": d["kernel_hash"],
                                 "phase": "compile"})
                continue
            meds, ntr = [], 0
            for _ in range(INVOCATIONS):
                r = h.run_kernel(str(exe))
                if not r.success:
                    continue
                trials = []
                for line in r.stdout.splitlines():
                    line = line.strip()
                    if line.startswith("{") and "time_ms" in line:
                        try:
                            trials.append(float(json.loads(line)["time_ms"]))
                        except (ValueError, KeyError):
                            pass
                if trials:
                    meds.append(statistics.median(trials))
                    ntr = len(trials)
            if len(meds) < 2:
                failures.append({"hash": d["kernel_hash"], "phase": "run",
                                 "error": f"only {len(meds)} invocations"})
                continue
            med = statistics.median(meds)
            row = {"kernel_hash": d["kernel_hash"], "op_type": d["op_type"],
                   "problem_size": d["problem_size"],
                   "launch_config": {"grid": d["grid"], "block": d["block"],
                                     "shared_mem_bytes": d["shared_mem_bytes"]},
                   "n_trials": ntr, "n_invocations": len(meds),
                   "median_runtime_ms": med,
                   "ai_cv": statistics.stdev(meds) / statistics.fmean(meds),
                   "invocation_medians_ms": meds, "protocol": PROTOCOL,
                   "queue_state": {
                       "own_job_id": q0.get("own_job_id"),
                       "n_other_active": q0.get("n_other_active")},
                   "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                   "measured_regime": ("launch_bound" if med < 0.010
                                       else "measurable")}
            rows.append(row)
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            if len(rows) % 10 == 0:
                print(f"GPQUEUE_PROGRESS {len(rows)} rows last={med:.4f} ms",
                      flush=True)
    cp.write_rows(HERE / "dose_measured.jsonl", rows)
    summary = {"bead": "gpu-sim-u2o.14", "n_specs": len(man["specs"]),
               "n_measured": len(rows), "n_failed": len(failures),
               "failures": failures[:10],
               "median_ai_cv": (statistics.median([r["ai_cv"] for r in rows])
                                if rows else None),
               "wall_sec": round(time.time() - t0, 1),
               "queue_at_start": {k: q0.get(k) for k in
                                  ("own_job_id", "n_other_active")}}
    (HERE / "dose_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"dose measured {len(rows)}/{len(man['specs'])} "
          f"failed={len(failures)} wall={summary['wall_sec']}s", flush=True)
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
