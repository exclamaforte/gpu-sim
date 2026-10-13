#!/usr/bin/env python3
"""gpu-sim-0vk MEASURE (GPU, through ts): prebuilt binaries, per-stage boundaries.

11.3's measure_batch shape with boundaries per STAGE (each stage's rows
share that stage's before-sample queue_state; the 6.6 contention rule runs
once per stage with the disposition reported -- run, not assumed). Rows with
fewer than 2 successful invocations are shortfall (named), never rows: every
shipped row carries ai_cv.

Row protocol is stamped here ("single_run_stages_2_5" -- the identical
procedure: one binary per kernel via kernel_driver.synthesize(), 2
invocations x 20 timed launches); corpus_protocol.write_rows refuses to
serialise an unstamped row. driver_scalars / manifest_occurrences are joined
at merge (merge_0vk.py). Provenance (bead item): every row carries
guard_passed_template=true -- the OOB guard passed on its template at build
(guard_verdicts.json); the 4 known-OOB frozen rows are named in REPORT.md,
not here, and stay frozen.

GPQUEUE_TRIAL=1 (submit-pair trial leg): first 30 specs only, binaries kept.
"""
import hashlib
import json
import statistics
import subprocess
import sys
import os
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from epic_u2o.safe_harness import SafeHarness  # noqa: E402
from epic_u2o.tokenizer import tokenize_for_model  # noqa: E402
from epic_u2o import queue_state as qs  # noqa: E402
from epic_u2o import corpus_protocol as cp  # noqa: E402
from epic_u2o.measurement_stats import apply_contention_rule  # noqa: E402

TRIALS = 20
INVOCATIONS = 2
MAX_TOKENS = 2048
PROTOCOL = "single_run_stages_2_5"

TELEMETRY_FIELDS = ["clocks.sm", "clocks.max.sm", "temperature.gpu",
                    "power.draw", "utilization.gpu",
                    "clocks_throttle_reasons.active", "persistence_mode"]


def telemetry():
    try:
        r = subprocess.run(
            ["nvidia-smi", f"--query-gpu={','.join(TELEMETRY_FIELDS)}",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30)
        if r.returncode == 0 and r.stdout.strip():
            return dict(zip(TELEMETRY_FIELDS,
                            [v.strip() for v in r.stdout.strip().split(",")]))
    except Exception:
        pass
    return {}


def queue_now():
    return qs.queue_state(own_description_substr="gpu-sim-0vk",
                          own_command_substr="measure_0vk")


def slim(q):
    return {k: q.get(k) for k in ("own_job_id", "n_other_active",
                                  "n_other_running_at_sample_time")}


_ELEMSIZE = {"float": 4, "unsigned int": 4, "int": 4, "double": 8}


def host_footprint_bytes(spec):
    """Host bytes the synthesised driver stages for one cell: every pointer
    arg is malloc'd + cudaMemcpy'd up front (PI part-2 diagnosis, read out of
    kernel_driver.synthesize). Pure; CPU-testable."""
    from epic_u2o.kernel_driver import _buffer_counts, _is_pointer
    dims = {k: v for k, v in (spec.get("metadata") or {}).items()
            if type(v) is int}
    n = spec["problem_size"]
    total = 0
    for a in spec.get("args", []):
        ctype = a.get("type", "")
        if not _is_pointer(ctype):
            continue
        base = ctype.replace("*", "").replace("__restrict__", "").replace(
            "const", "").strip()
        if base not in _ELEMSIZE:
            raise ValueError(f"unknown element type {ctype!r} for "
                             f"{spec.get('op_type')}")
        count = _buffer_counts({"op_type": spec.get("op_type")}, dims, n \
                               ).get(a["name"], n)
        total += count * _ELEMSIZE[base]
    return total


def mem_available_bytes():
    with open("/proc/meminfo") as fh:
        for line in fh:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("no MemAvailable in /proc/meminfo")


def preflight(spec, avail=None):
    """Per-cell availability gate (PI part-5 fix): refuse cells whose host
    footprint exceeds currently-available RAM, so an OOM-risk cell becomes a
    recorded skip instead of a fault indistinguishable from a template bug.
    Returns (ok, detail)."""
    need = host_footprint_bytes(spec)
    have = mem_available_bytes() if avail is None else avail
    if need > have:
        return (False, f"host footprint {need / 2 ** 30:.1f} GiB exceeds "
                       f"available {have / 2 ** 30:.1f} GiB")
    return (True, f"host footprint {need / 2 ** 30:.1f} GiB <= "
                   f"available {have / 2 ** 30:.1f} GiB")


def measure_stage(h, build, specs, fh, tel_rows, keep_binaries=False,
                    skip=frozenset()):
    stage = specs[0]["stage"]
    q0 = queue_now()
    tel0 = telemetry()
    print(f"[{stage}] {len(specs)} specs  queue={json.dumps(slim(q0))} "
          f"power={tel0.get('power.draw')} util={tel0.get('utilization.gpu')}",
          flush=True)
    rows, failures = [], []
    t_start = time.time()
    n_skipped = 0
    for i, d in enumerate(specs):
        if d["kernel_hash"] in skip:
            n_skipped += 1
            continue
        exe = build / f"k_{d['kernel_hash']}.out"
        if not exe.exists():
            failures.append({"hash": d["kernel_hash"], "phase": "compile",
                             "error": "no prebuilt binary; run compile_all.py"})
            continue
        ok, why = preflight(d)
        if not ok:
            failures.append({"hash": d["kernel_hash"],
                             "phase": "preflight-skip", "error": why})
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
                             "error": f"only {len(meds)} successful invocations; "
                                      "ai_cv needs 2 (row not shipped)"})
            continue
        med = statistics.median(meds)
        ai_cv = statistics.stdev(meds) / statistics.fmean(meds)
        toks = tokenize_for_model(d["source"], max_length=MAX_TOKENS)
        row = {
            "kernel_hash": d["kernel_hash"],
            "kernel_name": d["kernel_name"],
            "op_type": d.get("op_type"),
            "stage": stage,
            "problem_size": d["problem_size"],
            "launch_config": {"grid": d["grid"], "block": d["block"],
                              "shared_mem_bytes": d["shared_mem_bytes"],
                              "threads": d["grid"][0] * d["grid"][1] * d["grid"][2]
                              * d["block"][0] * d["block"][1] * d["block"][2]},
            "covers": d["covers"],
            "deliberately_poor": d["deliberately_poor"],
            "n_trials": ntr,
            "n_invocations": len(meds),
            "median_runtime_ms": med,
            "iqr_ms": (max(meds) - min(meds)) / 2.0,
            "ai_cv": ai_cv,
            "invocation_medians_ms": meds,
            "protocol": PROTOCOL,
            "guard_passed_template": True,
            "queue_state": {"own_job_id": q0.get("own_job_id"),
                            "n_other_active": q0.get("n_other_active"),
                            "n_other_running": q0.get(
                                "n_other_running_at_sample_time")},
            "token_count": len(toks),
            "token_digest": hashlib.sha256(
                ",".join(map(str, toks)).encode()).hexdigest()[:16],
            "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "measured_regime": ("launch_bound" if med < 0.010
                                else "measurable"),
        }
        if ai_cv > 0.15:
            row["unreliable"] = True
        rows.append(row)
        fh.write(json.dumps(row) + "\n")
        if len(rows) % 10 == 0:
            print(f"GPQUEUE_PROGRESS {len(rows)} rows {stage} "
                  f"last={med:.4f} ms", flush=True)
        if (i + 1) % 25 == 0:
            print(f"  [{stage}] {i + 1}/{len(specs)} last={med:.4f} ms",
                  flush=True)
        if not keep_binaries:
            exe.unlink(missing_ok=True)
    tel1 = telemetry()
    q1 = queue_now()
    disp = apply_contention_rule(tel0, tel1, q0, q1)
    tel_rows.append({"stage": stage, "n_measured": len(rows),
                     "n_failed": len(failures), "failures": failures[:10],
                     "wall_sec": time.time() - t_start,
                     "telemetry_before": tel0, "telemetry_after": tel1,
                     "queue_before": slim(q0), "queue_after": slim(q1),
                     "contention_disposition": disp})
    print(f"[{stage}] done {len(rows)}/{len(specs)} failed={len(failures)} "
          f"contention_fired={bool(disp.get('fired'))} "
          f"gaps={disp.get('coverage_gaps', disp.get('gaps', ''))}", flush=True)
    return rows, failures


def main() -> int:
    build = HERE / "build"
    manifest = json.loads((HERE / "manifest.json").read_text())
    specs = manifest["specs"]
    trial = os.environ.get("GPQUEUE_TRIAL") == "1"
    tag = os.environ.get("U0VK_TAG", "")
    skip = set()
    if os.environ.get("U0VK_SKIP"):
        for _sp in os.environ["U0VK_SKIP"].split(","):
            with open(_sp) as fh:
                for line in fh:
                    line = line.strip()
                    if line.startswith("{"):
                        skip.add(json.loads(line)["kernel_hash"])
    if skip:
        print(f"skipping {len(skip)} already-measured hashes", flush=True)
    if os.environ.get("U0VK_SLICE"):
        lo, hi = (int(x) for x in os.environ["U0VK_SLICE"].split(":"))
        specs = specs[lo:hi]
    if os.environ.get("U0VK_HASHES"):
        want = {l.strip() for l in open(os.environ["U0VK_HASHES"])
                if l.strip()}
        specs = [d for d in specs if d["kernel_hash"] in want]
    if trial:
        specs = specs[:30]
        suffix, keep = "_trial", True
    elif tag:
        suffix, keep = f"_{tag}", False
    else:
        suffix, keep = "", False
    by_stage = {}
    for d in specs:
        by_stage.setdefault(d["stage"], []).append(d)
    print(f"measuring {len(specs)} specs in {len(by_stage)} stages "
          f"({TRIALS} trials x {INVOCATIONS} invocations each) suffix={suffix!r}")
    h = SafeHarness(timeout_sec=600)
    all_rows, all_fail, stage_recs = [], [], []
    t0 = time.time()
    tmp = HERE / "build" / f"measured{suffix}_raw.jsonl"
    with tmp.open("w") as fh:
        for stage in sorted(by_stage):
            rows, failures = measure_stage(h, build, by_stage[stage], fh,
                                           stage_recs, keep, skip)
            all_rows.extend(rows)
            all_fail.extend(failures)
    cp.write_rows(HERE / f"measured{suffix}.jsonl", all_rows)
    summary = {
        "bead": "gpu-sim-0vk", "stage": "measure",
        "n_specs": len(specs), "n_measured": len(all_rows),
        "n_failed": len(all_fail),
        "n_launch_bound_measured": sum(1 for r in all_rows
                                       if r["measured_regime"] == "launch_bound"),
        "n_measurable_measured": sum(1 for r in all_rows
                                     if r["measured_regime"] == "measurable"),
        "n_unreliable": sum(1 for r in all_rows if r.get("unreliable")),
        "median_ai_cv": (statistics.median([r["ai_cv"] for r in all_rows])
                            if all_rows else None),
        "per_stage": stage_recs,
        "wall_sec": time.time() - t0,
        "rows_path": f"measured{suffix}.jsonl",
        "protocol": PROTOCOL,
        "trial": trial,
        "slice": os.environ.get("U0VK_SLICE", ""),
        "tag": tag,
    }
    (HERE / f"measure_summary{suffix}.json").write_text(json.dumps(summary, indent=2))
    print(f"\nmeasured {len(all_rows)}/{len(specs)} failed={len(all_fail)} "
          f"wall={summary['wall_sec']:.0f}s")
    return 0 if all_rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
