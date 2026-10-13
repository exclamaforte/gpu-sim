#!/usr/bin/env python3
"""gpu-sim-f6y MEASURE (GPU, through ts): prebuilt binaries, one stage boundary.

11.3's measure_batch shape, single stage (stage3): queue_state sampled before
the stage (own substrings name f6y, not u2o.11.3), before/after telemetry +
the 6.6 contention rule with the disposition reported -- run, not assumed.
Rows with fewer than 2 successful invocations are shortfall (named), never
rows: every shipped row carries ai_cv.

Row protocol is stamped here ("single_run_stages_2_5" -- the identical
procedure: one binary per kernel via kernel_driver.synthesize(), 2
invocations x 20 timed launches); corpus_protocol.write_rows refuses to
serialise an unstamped row, so a missing stamp fails loud, not silent.
driver_scalars / manifest_occurrences are joined at merge (merge_f6y.py),
same as 11.3's merge_slices.
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
    return qs.queue_state(own_description_substr="gpu-sim-f6y",
                          own_command_substr="measure_f6y")


def slim(q):
    return {k: q.get(k) for k in ("own_job_id", "n_other_active",
                                  "n_other_running_at_sample_time")}


def main() -> int:
    build = HERE / "build"
    specs = json.loads((HERE / "manifest.json").read_text())["specs"]
    assert len(specs) == 80, len(specs)
    assert {s["stage"] for s in specs} == {"stage3"}
    h = SafeHarness(timeout_sec=300)
    q0, tel0 = queue_now(), telemetry()
    print(f"[stage3] {len(specs)} specs queue={json.dumps(slim(q0))} "
          f"power={tel0.get('power.draw')} util={tel0.get('utilization.gpu')}",
          flush=True)
    rows, failures = [], []
    t0 = time.time()
    tmp = HERE / "build" / "measured_f6y_raw.jsonl"
    with tmp.open("w") as fh:
        for i, d in enumerate(specs):
            exe = build / f"k_{d['kernel_hash']}.out"
            if not exe.exists():
                failures.append({"hash": d["kernel_hash"], "phase": "compile",
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
            if len(meds) < 2:
                failures.append({"hash": d["kernel_hash"], "phase": "run",
                                 "error": f"only {len(meds)} successful "
                                          "invocations; ai_cv needs 2 (row not shipped)"})
                continue
            med = statistics.median(meds)
            ai_cv = statistics.stdev(meds) / statistics.fmean(meds)
            toks = tokenize_for_model(d["source"], max_length=MAX_TOKENS)
            row = {
                "kernel_hash": d["kernel_hash"],
                "kernel_name": d["kernel_name"],
                "op_type": d.get("op_type"),
                "stage": "stage3",
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
            print(f"  [stage3] {i + 1}/{len(specs)} {d['kernel_hash'][:8]} "
                  f"{d['op_type']} med={med:.4f} ms ai_cv={ai_cv:.4f}",
                  flush=True)
            exe.unlink(missing_ok=True)
    tel1, q1 = telemetry(), queue_now()
    disp = apply_contention_rule(tel0, tel1, q0, q1)
    wall = time.time() - t0
    cp.write_rows(HERE / "measured_f6y_raw_stamped.jsonl", rows)
    summary = {
        "bead": "gpu-sim-f6y", "stage": "measure",
        "n_specs": len(specs), "n_measured": len(rows),
        "n_failed": len(failures), "failures": failures[:10],
        "n_launch_bound_measured": sum(1 for r in rows
                                       if r["measured_regime"] == "launch_bound"),
        "n_unreliable": sum(1 for r in rows if r.get("unreliable")),
        "median_ai_cv": statistics.median([r["ai_cv"] for r in rows]) if rows else None,
        "wall_sec": wall,
        "telemetry_before": tel0, "telemetry_after": tel1,
        "queue_before": slim(q0), "queue_after": slim(q1),
        "contention_disposition": disp,
        "rows_path": "measured_f6y_raw_stamped.jsonl",
        "protocol": PROTOCOL,
    }
    (HERE / "measure_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nmeasured {len(rows)}/{len(specs)} failed={len(failures)} "
          f"wall={wall:.0f}s contention_fired={bool(disp.get('fired'))}")
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
