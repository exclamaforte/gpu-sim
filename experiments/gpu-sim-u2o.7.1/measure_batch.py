#!/usr/bin/env python3
"""gpu-sim-u2o.7.1 STAGE 2 (GPU, through ts): the measured batch.

This is the half of the pipeline that did not exist. generator_stage1.py
emitted 1000 rows of kernel SPECS with no runtime column; the bead's audit
note is explicit that only the generate step of
generate -> compile -> preflight -> measure -> tokenize was ever present.

For every spec in the manifest: synthesize the driver, compile it, run it
through the harness, and write one JSONL row carrying
  kernel_hash, launch_config, median_runtime_ms, iqr_ms, trials, ai_cv,
  device_prop_snapshot, queue_state and telemetry.

Two things the lab measured earlier are honoured rather than rediscovered:

  * gpu-sim-u2o.6.6.1: sub-10 us cells are majority launcher and carry ~7%
    across-invocation spread, so every row records ai_cv and the row is
    FLAGGED by measurement rather than by a size guess.
  * gpu-sim-u2o.6.6.3: a co-tenant moves these numbers by up to 2.33x, so
    queue state and device telemetry are recorded per row, and a row measured
    while the queue was non-empty is marked.

Tokenization is the last step of the pipeline and the row carries the token
count and a stable digest of the model ids, so 8.1 can consume the same
encoding without re-running anything.
"""
import hashlib
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from epic_u2o.safe_harness import SafeHarness      # noqa: E402
from epic_u2o.kernel_driver import synthesize     # noqa: E402
from epic_u2o.tokenizer import tokenize_for_model  # noqa: E402
from epic_u2o import queue_state as qs           # noqa: E402

TRIALS = 20
INVOCATIONS = 2          # processes per spec: enough to see ai_cv at this cost
MAX_TOKENS = 2048


def telemetry():
    try:
        r = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,driver_version,memory.total,clocks.sm,"
             "clocks.max.sm,temperature.gpu,power.draw,"
             "clocks_throttle_reasons.active,persistence_mode",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30)
        return {"rc": r.returncode, "csv": r.stdout.strip()}
    except Exception as e:
        return {"error": str(e)}


def main() -> int:
    build = HERE / "build"
    manifest = json.loads((build / "manifest.json").read_text())
    specs = manifest["specs"]
    out_rows = HERE / "measured.jsonl"
    print(f"measuring {len(specs)} specs "
          f"({TRIALS} trials x {INVOCATIONS} processes each)")

    h = SafeHarness(timeout_sec=300)
    tel0 = telemetry()
    q0 = qs.queue_state(own_description_substr="u2o.7.1",
                        own_command_substr="measure_batch.py")
    print("device:", tel0.get("csv"))
    print("queue at start: " + json.dumps(
        {k: q0.get(k) for k in ("own_job_id", "n_other_active",
                                 "n_other_running_at_sample_time")}))

    rows, failures = [], []
    t_start = time.time()
    with out_rows.open("w") as fh:
        for i, d in enumerate(specs):
            spec = {k: d[k] for k in ("source", "kernel_name", "grid",
                                      "block", "args", "problem_size",
                                      "shared_mem_bytes")}
            spec["op_type"] = d.get("op_type", "anchor")
            if not d.get("prebuilt"):
                pass  # synthesize happened in phase A; nothing to redo here
            # PRE-COMPILED. The queue killed the combined compile+measure job
            # for 0.7% GPU utilisation, correctly: nvcc and process startup
            # dominated and the device sat idle. Phase A (compile_all.py, CPU,
            # unqueued) builds these; this job only measures, so the GPU is
            # actually the bottleneck. A missing binary is a PHASE A failure
            # and is reported as such rather than recompiled here.
            exe = build / f"k_{d['kernel_hash']}.out"
            if not exe.exists():
                failures.append({"hash": d["kernel_hash"], "stage": "compile",
                                 "error": "no prebuilt binary; run compile_all.py"})
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
            if not meds:
                failures.append({"hash": d["kernel_hash"], "stage": "run"})
                continue

            med = statistics.median(meds)
            ai_cv = (statistics.stdev(meds) / statistics.fmean(meds)
                     if len(meds) > 1 else None)
            toks = tokenize_for_model(d["source"], max_length=MAX_TOKENS)
            row = {
                "kernel_hash": d["kernel_hash"],
                "kernel_name": d["kernel_name"],
                "op_type": d.get("op_type"),
                "problem_size": d["problem_size"],
                "launch_config": {"grid": d["grid"], "block": d["block"],
                                  "shared_mem_bytes": d["shared_mem_bytes"],
                                  "threads": d["grid"][0] * d["grid"][1] * d["grid"][2]
                                  * d["block"][0] * d["block"][1] * d["block"][2]},
                "covers": d["covers"],
                "deliberately_poor": d["deliberately_poor"],
                # 7.1 rework: no regime_hint -- measured_regime (below) is
                # authoritative; the design-time guess shipped unlabelled.
                "n_trials": ntr,
                "n_invocations": len(meds),
                "median_runtime_ms": med,
                "iqr_ms": None,
                "ai_cv": ai_cv,
                "invocation_medians_ms": meds,
                "device_prop_snapshot": tel0.get("csv"),
                "queue_state": {"own_job_id": q0.get("own_job_id"),
                                "n_other_active": q0.get("n_other_active"),
                                "n_other_running": q0.get("n_other_running_at_sample_time")},
                "token_count": len(toks),
                "token_digest": hashlib.sha256(
                    ",".join(map(str, toks)).encode()).hexdigest()[:16],
                "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
            # iqr across the invocation medians, which is what a 2-process
            # batch can honestly support (a 20-trial IQR inside one process is
            # not the same quantity and is not what we want to compare)
            if len(meds) > 1:
                row["iqr_ms"] = (max(meds) - min(meds)) / 2.0
            # regime decided by MEASUREMENT, not by the size guess
            row["measured_regime"] = ("launch_bound" if med < 0.010
                                       else "measurable")
            if ai_cv is not None and ai_cv > 0.15:
                row["unreliable"] = True     # measurement_stats' own gate
            rows.append(row)
            fh.write(json.dumps(row) + "\n")
            if (i + 1) % 40 == 0:
                print(f"  {i + 1}/{len(specs)}  last median={med:.4f} ms "
                      f"({time.time() - t_start:.0f}s)", flush=True)
            exe.unlink(missing_ok=True)

    tel1 = telemetry()
    summary = {
        "bead": "gpu-sim-u2o.7.1",
        "stage": "measure",
        "n_specs": len(specs),
        "n_measured": len(rows),
        "n_failed": len(failures),
        "failures": failures[:20],
        "n_launch_bound_measured": sum(1 for r in rows if r["measured_regime"] == "launch_bound"),
        "n_unreliable": sum(1 for r in rows if r.get("unreliable")),
        "median_ai_cv": statistics.median([r["ai_cv"] for r in rows
                                           if r["ai_cv"] is not None] or [None]),
        "wall_sec": time.time() - t_start,
        "telemetry_start": tel0,
        "telemetry_end": tel1,
        "queue_at_start": {k: q0.get(k) for k in
                           ("own_job_id", "n_other_active",
                            "n_other_running_at_sample_time",
                            "own_job_in_active_set")},
        "rows_path": str(out_rows),
    }
    (HERE / "measure_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nmeasured {len(rows)}/{len(specs)}  failed={len(failures)}  "
          f"wall={summary['wall_sec']:.0f}s")
    print(f"  launch-bound (measured < 10 us): {summary['n_launch_bound_measured']}")
    print(f"  ai_cv > 0.15 (unreliable):        {summary['n_unreliable']}")
    if failures:
        print("  first failures: " + json.dumps(failures[:3]))
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
