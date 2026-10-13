#!/usr/bin/env python3
"""gpu-sim-u2o.6.7.1 follow-up: the missing 2x2 corner.

7.1's OWN per-kernel binaries under 7.1's OWN per-invocation process model:
for each spec, compile k_<hash>.out via kernel_driver.synthesize() exactly as
7.1's compile_all.py does (own main, own cudaMalloc, baked geometry -- NOT the
hoisted-geometry objects of perkernel_confound.py), then measure with 7.1's
own loop (INVOCATIONS=2 separate run_kernel processes per spec, TRIALS=20,
median of per-invocation medians). Same estimator, same examples.

If this reproduces 7.1's medians, the fifth mechanism is the shared build
context and gets a name; if not, the difference lives in run_kernel()/the
preexec limits and the context is exonerated. Both outcomes decision-grade
(PI design call 2026-10-04).

Two phases, split where the queue policy splits them:
  --phase build   CPU, unqueued: regenerate specs, PROVE them 7.1's by hash
                  intersection (refuse otherwise), compile all binaries.
  --phase measure GPU, one ts job: the ~70s measurement loop.
Both arms (this corner and 7.1's reference) run on the same tree, so the 6.3
run_kernel restructure is held constant across the comparison by construction.
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
sys.path.insert(0, str(HERE))

from epic_u2o.safe_harness import SafeHarness  # noqa: E402
from epic_u2o.kernel_driver import synthesize  # noqa: E402
from perkernel_confound import load_specs  # noqa: E402 -- 7.1's specs + hash proof

ARCH = "sm_120"
TRIALS_EXPECTED = 20
INVOCATIONS = 2
BUILD_DIR = HERE / "corner_build"
ROWS = HERE / "rows_corner.jsonl"

# Same tolerance rule as verify_reproduction.py: per-row, from the reference
# row's own ai_cv, floored and capped. A global bar would repeat the mistake
# 9.4's criterion 6 was written to prevent (cells differ).
TOL_FLOOR, TOL_CAP, TOL_SIGMA = 0.10, 0.75, 3.0

REFERENCE_REF = ("lab/gpu-sim-u2o.7.1:"
                 "experiments/gpu-sim-u2o.7.1/measured.jsonl")


def reference_medians() -> dict:
    r = subprocess.run(["git", "show", REFERENCE_REF], cwd=str(REPO),
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"FATAL: cannot read {REFERENCE_REF}")
    return {json.loads(line)["kernel_hash"]: json.loads(line)
            for line in r.stdout.splitlines() if line.strip()}


def telemetry():
    try:
        r = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,power.draw,temperature.gpu",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30)
        return {"rc": r.returncode, "csv": r.stdout.strip()}
    except Exception as e:
        return {"error": str(e)}


def queue_sample(tag: str) -> dict:
    from epic_u2o import queue_state as qs
    q = qs.queue_state(own_description_substr="u2o.6.7.1",
                       own_command_substr="corner_71x71.py")
    return {"tag": tag,
            "own_job_id": q.get("own_job_id"),
            "n_other_active": q.get("n_other_active"),
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}


def phase_build() -> int:
    specs, proof = load_specs()
    print(f"specs: {len(specs)}, proof={proof}")
    if not proof.get("proven_identical"):
        print("FATAL: specs not proven identical to 7.1's "
              "(hash-intersection refusal -- a different sweep would "
              "reproduce the confound while claiming to remove it).",
              flush=True)
        return 2
    BUILD_DIR.mkdir(exist_ok=True)
    h = SafeHarness(timeout_sec=300)
    ok = bad = 0
    errors = []
    t0 = time.time()
    for i, s in enumerate(specs):
        spec = {k: s[k] for k in ("source", "kernel_name", "grid", "block",
                                  "args", "problem_size", "shared_mem_bytes")}
        spec["op_type"] = s.get("op_type", "anchor")
        try:
            program = synthesize(spec)
        except Exception as e:
            bad += 1
            errors.append({"hash": s["kernel_hash"], "stage": "synthesize",
                           "error": str(e)[:200]})
            continue
        c, m = h.compile_kernel(program,
                                str(BUILD_DIR / f"k_{s['kernel_hash']}.out"),
                                arch=ARCH)
        if c:
            ok += 1
        else:
            bad += 1
            errors.append({"hash": s["kernel_hash"], "stage": "compile",
                           "error": str(m)[:200]})
        if (i + 1) % 50 == 0:
            print(f"  compiled {i + 1}/{len(specs)} ({time.time() - t0:.0f}s)",
                  flush=True)
    (HERE / "corner_build_report.json").write_text(json.dumps({
        "n_specs": len(specs), "n_compiled": ok, "n_failed": bad,
        "proof": proof, "errors": errors[:20]}, indent=2))
    print(f"compiled {ok}/{len(specs)}, {bad} failed")
    return 0 if ok and not bad else 1


def phase_measure() -> int:
    specs, proof = load_specs()
    if not proof.get("proven_identical"):
        print("FATAL: hash-intersection refusal.", flush=True)
        return 2
    ref = reference_medians()
    h = SafeHarness(timeout_sec=300)
    q0 = queue_sample("start")
    tel0 = telemetry()
    print(f"measuring {len(specs)} specs "
          f"({TRIALS_EXPECTED} trials x {INVOCATIONS} processes each)")
    print(f"queue at start: own={q0['own_job_id']} "
          f"n_other={q0['n_other_active']}")
    rows, failures = [], []
    t_start = time.time()
    with ROWS.open("w") as fh:
        for i, s in enumerate(specs):
            exe = BUILD_DIR / f"k_{s['kernel_hash']}.out"
            if not exe.exists():
                failures.append({"hash": s["kernel_hash"], "stage": "compile",
                                 "error": "no Phase-A binary"})
                continue
            meds = []
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
            if not meds:
                failures.append({"hash": s["kernel_hash"], "stage": "run"})
                continue
            med = statistics.median(meds)
            ai_cv = (statistics.stdev(meds) / statistics.fmean(meds)
                     if len(meds) > 1 else None)
            row = {"kernel_hash": s["kernel_hash"],
                   "median_runtime_ms": med,
                   "n_invocations": len(meds),
                   "invocation_medians_ms": meds,
                   "ai_cv": ai_cv,
                   "queue_state": queue_sample(f"row-{i}"),
                   "telemetry": telemetry()}
            rows.append(row)
            fh.write(json.dumps(row) + "\n")
            if (i + 1) % 40 == 0:
                print(f"  {i + 1}/{len(specs)} last={med:.4f} ms "
                      f"({time.time() - t_start:.0f}s)", flush=True)
    wall = time.time() - t_start
    # Same comparison as verify_reproduction.py, plus the requested columns.
    rels, reproducing, faster, aicvs, worst = [], 0, 0, [], None
    for r in rows:
        b = ref.get(r["kernel_hash"])
        if not b:
            continue
        base = b["median_runtime_ms"]
        rel = abs(r["median_runtime_ms"] - base) / base
        tol = max(TOL_FLOOR, min(TOL_SIGMA * (b.get("ai_cv") or 0.0), TOL_CAP))
        rels.append(rel)
        if rel <= tol:
            reproducing += 1
        elif worst is None or rel > worst[0]:
            worst = (rel, r["kernel_hash"])
        if r["median_runtime_ms"] < base:
            faster += 1
        if r["ai_cv"] is not None:
            aicvs.append(r["ai_cv"])
    summary = {
        "bead": "gpu-sim-u2o.6.7.1",
        "arm": "corner_71x71: 7.1 binaries x 7.1 process model",
        "n_specs": len(specs), "n_measured": len(rows),
        "n_failed": len(failures), "failures": failures[:10],
        "n_reference": len(ref),
        "reproduce_count": reproducing,
        "reproduce_fraction": round(reproducing / len(rels), 4) if rels else None,
        "median_rel_diff": round(statistics.median(rels), 4) if rels else None,
        "faster_fraction": round(faster / len(rels), 4) if rels else None,
        "median_ai_cv": round(statistics.median(aicvs), 4) if aicvs else None,
        "worst": {"rel": worst[0], "hash": worst[1]} if worst else None,
        "ex_per_s": round(len(rows) / wall, 2),
        "wall_s": round(wall, 1),
        "queue_at_start": q0, "telemetry_at_start": tel0,
        "tree_rev": subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True).stdout.strip(),
    }
    (HERE / "corner_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


def main() -> int:
    if len(sys.argv) != 3 or sys.argv[1] != "--phase" \
            or sys.argv[2] not in ("build", "measure"):
        print("usage: corner_71x71.py --phase build|measure", file=sys.stderr)
        return 2
    return phase_build() if sys.argv[2] == "build" else phase_measure()


if __name__ == "__main__":
    raise SystemExit(main())
