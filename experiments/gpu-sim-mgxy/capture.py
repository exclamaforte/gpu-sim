#!/usr/bin/env python3
"""Capture gpu-sim-mgxy's null-cell measurement from ONE gpqueue job.

Reads the run directory run.sh printed, copies that invocation's raw per-repeat
log to output.log, reads summary.json, and emits metrics.json + job.json with
the job's own queue record.  A missing or malformed value fails the capture.
"""

import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
GPQUEUE = Path("/home/gabe/Documents/gpqueue/gpqueue.py")
JOBS = Path("/home/gabe/.local/state/gpqueue/jobs")


def main(argv):
    if len(argv) != 3:
        raise SystemExit("usage: capture.py <run dir> <gpqueue job id>")
    run_dir, job_id = Path(argv[1]).resolve(), argv[2]
    summary = json.loads((run_dir / "summary.json").read_text())
    raw = run_dir / "null_cell.log"
    if not raw.exists():
        raise SystemExit(f"missing {raw}")
    for key in ("n_repeats", "null_median", "null_warm_min", "real_median",
                "real_warm_min", "wall_s"):
        if key not in summary:
            raise SystemExit(f"summary missing {key!r}")
    if summary["n_repeats"] != summary["requested_repeats"]:
        raise SystemExit("fewer repeats parsed than requested")

    status = json.loads(subprocess.run(
        [sys.executable, str(GPQUEUE), "status", "--json"],
        capture_output=True, text=True, check=True).stdout)
    job = next((j for j in status if j["id"] == job_id), None)
    if job is None:
        raise SystemExit(f"job {job_id} not in gpqueue status")
    if job.get("exit_code") != 0:
        raise SystemExit(f"job {job_id} exited {job.get('exit_code')}")

    shutil.copyfile(raw, HERE / "output.log")
    tip = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                         capture_output=True, text=True, check=True).stdout.strip()
    sources = {n: __import__("hashlib").sha256((HERE / n).read_bytes()).hexdigest()
               for n in ("run.sh", "capture.py", "instrument_null.py")}

    metrics = {
        "bead": "gpu-sim-mgxy",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "tip": tip,
        "scope": (
            "The NULL cell, re-measured at this tip on the real RTX 5090 through "
            "the GPU queue. The SAME empty binary is run twice and treated as "
            "(scale, empty): a null cell cannot be faster than itself, so any "
            "negative kernel_us there is the estimator's own noise. Reported per "
            "estimator over 20 repeats: how many readings are negative. No "
            "admission-threshold change and no kernel_us clamping is claimed."
        ),
        "substituted_input": {
            "declared": (
                "The provenance log used prebuilt 8zqr binaries including "
                "empty_16_1_1_64_1_1.out (grid 16x64). That build directory is not "
                "in the tree; building the divergent_branch family regenerates the "
                "baseline as grid 4x64, which is what this arm uses."
            ),
            "empty_binary": "experiments/gpu-sim-8zqr/build/bin/empty_4_1_1_64_1_1.out",
            "scale_binary": "experiments/gpu-sim-8zqr/build/bin/k_divergent_branch_0000.out",
            "consequence": (
                "The RATE of negative null readings is baseline- and run-dependent: "
                "the prebuilt 16x64 pair gave 10/20 in the provenance log, this arm "
                "measured 3/20 in one run at the old tip and 7/20 at this tip. The "
                "SIGN defect is present in every run; the exact count is not "
                "reproducible and is not asserted."
            ),
        },
        "result": {
            "n_repeats": summary["n_repeats"],
            "null_median": summary["null_median"],
            "null_warm_min": summary["null_warm_min"],
            "real_median": summary["real_median"],
            "real_warm_min": summary["real_warm_min"],
            "wall_s": summary["wall_s"],
        },
        "reading": {
            "finding": (
                "The shipped median estimator returns a NEGATIVE kernel_us for a "
                "cell that is the same binary as its own floor, so a negative "
                "reading is the estimator's noise and carries no information about "
                "the kernel. The warm-min floor removes nearly all of them; the "
                "real tiny kernel stays non-negative."
            ),
            "not_claimed": (
                "No claim that any specific row's kernel_us is correct, no change to "
                "the admission threshold, and the negative is never clamped to 0 "
                "(7m60's flagging stays the safety net)."
            ),
        },
        "evidence": {
            "run_sh": "experiments/gpu-sim-mgxy/run.sh",
            "capture_py": "experiments/gpu-sim-mgxy/capture.py",
            "instrument_py": "experiments/gpu-sim-mgxy/instrument_null.py",
            "output_log": "experiments/gpu-sim-mgxy/output.log",
            "job_json": "experiments/gpu-sim-mgxy/job.json",
            "arm_sources_sha256": sources,
            "run_dir": str(run_dir),
        },
        "gate": "cited in the submission comment; no gate is claimed here",
    }

    job_json = {
        "bead": "gpu-sim-mgxy",
        "gpu_used": True,
        "job_id": job_id,
        "description": job.get("description"),
        "gpu_index": job.get("gpu_index"),
        "command": job.get("command"),
        "started_at_utc": job.get("started_at"),
        "finished_at_utc": job.get("finished_at"),
        "wall_s": job.get("wall_s"),
        "state": job.get("state"),
        "exit_code": job.get("exit_code"),
        "job_log": str(JOBS / f"{job_id}.log"),
        "run_dir": str(run_dir),
        "run_sh": "experiments/gpu-sim-mgxy/run.sh",
        "capture_py": "experiments/gpu-sim-mgxy/capture.py",
    }

    (HERE / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (HERE / "job.json").write_text(json.dumps(job_json, indent=2) + "\n")
    print(json.dumps({"tip": tip, "job": job_id,
                      "null_median_negative": summary["null_median"]["negative"],
                      "null_warm_min_negative": summary["null_warm_min"]["negative"],
                      "real_warm_min_negative": summary["real_warm_min"]["negative"]},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
