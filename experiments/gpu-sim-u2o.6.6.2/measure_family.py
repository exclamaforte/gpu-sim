#!/usr/bin/env python3
"""Measure ONE anchor family on the real 5090 (gpu-sim-u2o.6.6.2).

One ts job per family, so a failure costs a family and not the sweep.

Path: kernel_driver.synthesize -> compile_kernel -> run_kernel, i.e. exactly
the measurement path the 6.1.3/6.1.1 rework is NOT supposed to touch, so that
6.1.1's post-rework closing run can be diffed against this anchor.
preflight_check is BYPASSED on purpose: it executes on the real GPU (PI ruling
on 6.1.1) and an unaccounted second execution is exactly what this anchor must
not inherit. Read-only nvidia-smi telemetry before and after; if throttling
appears it is recorded per config rather than averaged away.
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
sys.path.insert(0, str(HERE))

from epic_u2o.kernel_driver import synthesize  # noqa: E402
from epic_u2o.safe_harness import SafeHarness  # noqa: E402
from epic_u2o import queue_state as qs  # noqa: E402
import anchor_kernels  # noqa: E402

OWN_JOB_DESCRIPTION = os.environ.get("U2O_JOB_DESC", "u2o.6.6.2 anchor")

TRIALS = 100
INVOCATIONS = 3          # independent PROCESSES, not trials
TELEMETRY_FIELDS = ["clocks.sm", "clocks.max.sm", "temperature.gpu",
                    "clocks_throttle_reasons.active", "persistence_mode"]


def telemetry(label):
    out = {"label": label, "at": time.time()}
    try:
        r = subprocess.run(
            ["nvidia-smi", f"--query-gpu={','.join(TELEMETRY_FIELDS)}",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=30)
        out["rc"] = r.returncode
        out["csv"] = r.stdout.strip()
        if r.returncode == 0 and r.stdout.strip():
            out["fields"] = dict(zip(TELEMETRY_FIELDS,
                                     [v.strip() for v in r.stdout.strip().split(",")]))
    except Exception as e:
        out["error"] = str(e)
    return out


def throttle_active(tel):
    f = (tel or {}).get("fields", {})
    v = f.get("clocks_throttle_reasons.active", "")
    return v not in ("", "0x0000000000000000", "Not Supported")


def parse_trials(stdout):
    out = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if "time_ms" in rec:
            out.append(float(rec["time_ms"]))
    return out


def measure_family(family: str) -> dict:
    print("=" * 72)
    print(f"gpu-sim-u2o.6.6.2 anchor -- family '{family}'")
    print("=" * 72)
    print("PREFLIGHT IS BY-PASSED ON PURPOSE: preflight_check() executes on the")
    print("real GPU (PI ruling on 6.1.1); calling it would add a second,")
    print("unaccounted GPU execution to an anchor that exists precisely to be")
    print("diffed against a later run. synthesize -> compile_kernel -> run_kernel")
    print("only. Kernels are hand-written (anchor_kernels.py), not generated:")
    print("gpu-sim-9ti showed stage2-5 kernel_name is untrustworthy.")
    print("No sudo, no clock locking, no persistence-mode change.", flush=True)

    anchor_kernels.self_check()
    tel_before = telemetry("before")
    print("telemetry before: " + json.dumps(tel_before.get("fields", tel_before)),
          flush=True)

    # Item 8 / the PI's standing rule: record the queue state and any
    # overlapping JOB_IDs at measurement time, or the artifact is unauditable
    # and must be treated as unverified. This is the matched half of 6.1.1's
    # item 6: the three-axis diff needs BOTH sides to carry it. Record only --
    # no contention detection and no sample discarded on the basis of this
    # (that is 6.6.3's question).
    # Match on the COMMAND plus the family argument, not on a hand-written
    # description substring. The v7 run recorded a stale own_job_id because
    # U2O_JOB_DESC said "anchor: memory" while the description actually
    # submitted to ts was "anchor v7: memory", so the description match found
    # only an OLDER job with the same command. The family's own argument is in
    # the command line, so matching on it is exact and needs no agreement
    # between a string I write here and a string typed at submit time.
    q_before = qs.queue_state(own_description_substr=OWN_JOB_DESCRIPTION,
                              own_command_substr=f"run_family.sh {family}")
    print("queue before: " + json.dumps(
        {k: q_before[k] for k in ("own_job_id", "n_other_active",
                                   "n_other_running_at_sample_time")}),
        flush=True)
    print("overlapping jobs: " + json.dumps(
        [{"id": j["id"], "state": j["state"], "description": j["description"]}
         for j in q_before["other_active_jobs"]]), flush=True)

    harness = SafeHarness(timeout_sec=180, vram_limit_mb=24576)
    configs = anchor_kernels.configs(family)
    print(f"{family}: {len(configs)} configs x {INVOCATIONS} process "
          f"invocations x {TRIALS} trials", flush=True)

    results = []
    t0 = time.time()
    for i, (spec, label) in enumerate(configs):
        try:
            program = synthesize(spec)
        except Exception as e:
            results.append({"label": label, "error": f"synthesize: {e}"})
            print(f"  !! {label}: synthesize failed: {e}", flush=True)
            continue
        exe = HERE / f"anchor_{family}_{i}.out"
        ok, msg = harness.compile_kernel(program, str(exe), arch="sm_120")
        if not ok:
            results.append({"label": label, "error": f"compile: {msg[:300]}"})
            print(f"  !! {label}: compile failed: {str(msg)[:160]}", flush=True)
            continue

        per_inv = []
        for inv in range(INVOCATIONS):
            res = harness.run_kernel(str(exe))
            if not res.success:
                print(f"  !! {label} inv{inv}: rc={res.returncode} "
                      f"{res.stderr[:160]!r}", flush=True)
                continue
            trials = parse_trials(res.stdout)
            if not trials:
                print(f"  !! {label} inv{inv}: no trial JSON", flush=True)
                continue
            per_inv.append(trials)

        exe.unlink(missing_ok=True)
        if not per_inv:
            results.append({"label": label, "error": "no successful invocations"})
            continue

        meds = [statistics.median(t) for t in per_inv]
        med = statistics.median(meds)
        # across-invocation CV of the MEDIAN: the number that decides whether
        # any of this is reproducible, and the one job 9c2420b3 never measured
        aicv = (statistics.stdev(meds) / statistics.fmean(meds)
                if len(meds) > 1 and statistics.fmean(meds) else None)
        pooled = sorted(x for t in per_inv for x in t)
        q1, q3 = pooled[len(pooled) // 4], pooled[3 * len(pooled) // 4]
        mean = statistics.fmean(pooled)
        sd = statistics.stdev(pooled) if len(pooled) > 1 else 0.0
        tel_now = telemetry(f"cfg{i}") if i % 5 == 0 else None
        rec = {
            "label": label,
            "kernel": spec["kernel_name"],
            "op_type": spec["op_type"],
            "n_problem_size": spec["problem_size"],
            "grid": spec["grid"],
            "block": spec["block"],
            "shared_mem_bytes": spec["shared_mem_bytes"],
            "n_trials": TRIALS,
            "n_invocations": len(per_inv),
            "median_ms": med,
            "iqr_ms": q3 - q1,
            "mean_ms": mean,
            "std_ms": sd,
            "cv": (sd / mean) if mean else 0.0,
            "medians_across_invocations_ms": meds,
            "across_invocation_cv": aicv,
            "throttle_active": throttle_active(tel_now or tel_before),
        }
        results.append(rec)
        print(f"  {label:28s} grid={str(spec['grid']):>14s} "
              f"med={med*1000:9.3f}us iqr={(q3-q1)*1000:8.3f}us "
              f"cv={rec['cv']:.3f} ai_cv={aicv if aicv is None else round(aicv,4)}",
              flush=True)

    tel_after = telemetry("after")
    print("telemetry after:  " + json.dumps(tel_after.get("fields", tel_after)),
          flush=True)
    q_after = qs.queue_state(own_description_substr=OWN_JOB_DESCRIPTION,
                             own_command_substr=f"run_family.sh {family}")
    return {
        "bead": "gpu-sim-u2o.6.6.2",
        "family": family,
        "queue_state_before": q_before,
        "queue_state_after": q_after,
        "preflight_bypassed": True,
        "preflight_bypass_reason": ("preflight_check executes on the real GPU "
                                    "(PI ruling 6.1.1); the anchor exists to be "
                                    "diffed against a later run and must not "
                                    "carry an unaccounted extra execution"),
        "telemetry_before": tel_before,
        "telemetry_after": tel_after,
        "throttling_observed": (throttle_active(tel_before) or throttle_active(tel_after)),
        "design": {"trials": TRIALS, "invocations": INVOCATIONS,
                   "note": "invocations are independent PROCESSES"},
        "results": results,
        "wall_sec": time.time() - t0,
    }


def main() -> int:
    family = sys.argv[1] if len(sys.argv) > 1 else "launch"
    data = measure_family(family)
    out = HERE / f"family_{family}.json"
    out.write_text(json.dumps(data, indent=2))
    print(f"\nwrote {out}")
    ok = sum(1 for r in data["results"] if "error" not in r)
    print(f"{family}: {ok}/{len(data['results'])} configs measured, "
          f"{data['wall_sec']:.1f}s")
    return 0 if ok == len(data["results"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
