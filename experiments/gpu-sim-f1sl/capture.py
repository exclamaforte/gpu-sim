#!/usr/bin/env python3
"""Capture f1sl: run the two tests, require 8 passed / 0 skipped, write artifacts."""
import json
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RUN = HERE / "run.sh"
OUT = HERE / "output.log"
METRICS = HERE / "metrics.json"
JOB = HERE / "job.json"


def main():
    started = datetime.now(timezone.utc).isoformat()
    t0 = time.monotonic()
    replay = REPO / ".lab" / "tmp" / "f1sl-replay.log"
    replay.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["bash", str(RUN), str(replay)],
        cwd=str(REPO),
        capture_output=True,
        text=True,
    )
    wall = round(time.monotonic() - t0, 1)
    text = replay.read_text() if replay.exists() else ""
    OUT.write_text(text)
    passed = re.search(r"(\d+) passed", text)
    skipped = re.search(r"(\d+) skipped", text)
    failed = re.search(r"(\d+) failed", text)
    n_passed = int(passed.group(1)) if passed else 0
    n_skipped = int(skipped.group(1)) if skipped else 0
    n_failed = int(failed.group(1)) if failed else 0
    source_count = sum(
        (REPO / rel).read_text().count("cannot create scratch worktree:")
        for rel in (
            "tests/test_verify_committed_run.py",
            "tests/test_zsd3_janitor_detection.py",
        )
    )
    result = {
        "exit_code": proc.returncode,
        "tests_passed": n_passed,
        "tests_skipped": n_skipped,
        "tests_failed": n_failed,
        "worktree_skip_sites_remaining": source_count,
        "skip_allowlist_entries_remaining": (
            (REPO / "tests" / "skip_allowlist.json").read_text().count(
                "cannot create scratch worktree"
            )
        ),
        "checks": {
            "replay_exit_zero": proc.returncode == 0,
            "eight_passed": n_passed == 8,
            "zero_skipped": n_skipped == 0,
            "zero_failed": n_failed == 0,
            "no_skip_sites_in_source": source_count == 0,
            "no_skip_allowlist_entry": (
                (REPO / "tests" / "skip_allowlist.json").read_text().count(
                    "cannot create scratch worktree"
                ) == 0
            ),
        },
    }
    metrics = {
        "bead": "gpu-sim-f1sl",
        "captured_at_utc": started,
        "scope": (
            "CPU-only test fix: two scratch-worktree fixtures now fail loudly "
            "with the git error instead of pytest.skip; both files also prove "
            "the failure path with a monkeypatched worktree-add failure."
        ),
        "gate": (
            "none -- CPU-only test fix; the final gate result is recorded in "
            "the submission comment"
        ),
        "result": result,
        "evidence": {
            "run_sh": "experiments/gpu-sim-f1sl/run.sh",
            "capture_py": "experiments/gpu-sim-f1sl/capture.py",
            "output_log": "experiments/gpu-sim-f1sl/output.log",
            "job_json": "experiments/gpu-sim-f1sl/job.json",
        },
    }
    job = {
        "bead": "gpu-sim-f1sl",
        "gpu_used": False,
        "cpu_only": True,
        "why_no_gpu": "test-fixture bug; no CUDA and no GPU work",
        "cpu_runs": [{
            "command": "bash experiments/gpu-sim-f1sl/run.sh",
            "exit_code": proc.returncode,
            "log": "experiments/gpu-sim-f1sl/output.log",
            "what": "two worktree fixtures: 8 passed, 0 skipped, failure path proved",
            "wall_s": wall,
            "started_at_utc": started,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        }],
    }
    METRICS.write_text(json.dumps(metrics, indent=2) + "\n")
    JOB.write_text(json.dumps(job, indent=2) + "\n")
    print(json.dumps({"exit_code": proc.returncode, **result["checks"]}, indent=2))
    return 0 if all(result["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
