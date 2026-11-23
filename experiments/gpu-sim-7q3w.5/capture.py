#!/usr/bin/env python3
"""Capture gpu-sim-7q3w.5's first experiment from the actual gpqueue job.

The job ran on the real GPU queue; this script copies that job's log verbatim
to output.log, reads the per-rank fabric counters the run wrote, and emits
metrics.json + job.json.  It does not synthesize measurements: every value
below is parsed from the job log or those counter files, and a missing/malformed
value fails the capture instead of being defaulted.
"""
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
JOB_ID = "20261123t000811-190b5e6c"
RUN_ID = "run-20261122T160811"
JOB_LOG = Path("/home/gabe/.local/state/gpqueue/jobs") / f"{JOB_ID}.log"
RUN_DIR = REPO / ".lab" / "tmp" / "build-7q3w5-numerics" / RUN_ID
OUT = HERE / "output.log"
METRICS = HERE / "metrics.json"
JOB = HERE / "job.json"
GPQUEUE = Path("/home/gabe/Documents/gpqueue/gpqueue.py")


def job_status():
    cp = subprocess.run([sys.executable, str(GPQUEUE), "status", "--json"],
                        capture_output=True, text=True, check=True)
    for job in json.loads(cp.stdout):
        if job["id"] == JOB_ID:
            return job
    raise SystemExit(f"job {JOB_ID} not in gpqueue status")


def main():
    if not JOB_LOG.exists():
        raise SystemExit(f"job log missing: {JOB_LOG}")
    if not RUN_DIR.is_dir():
        raise SystemExit(f"run dir missing: {RUN_DIR}")
    text = JOB_LOG.read_text()
    OUT.write_text(text)

    allreduce = {}
    for m in re.finditer(
        r"^NUMERICS allreduce rank=(\d+) n=(\d+) expected=([\d.]+) "
        r"value=([\d.]+) max_abs_err=([\d.eE+-]+) bad=(\d+)$",
        text, re.M):
        rank = int(m.group(1))
        allreduce[rank] = {
            "n": int(m.group(2)),
            "expected": float(m.group(3)),
            "value": float(m.group(4)),
            "max_abs_err": float(m.group(5)),
            "bad": int(m.group(6)),
        }
    if set(allreduce) != {0, 1}:
        raise SystemExit(f"expected allreduce lines for ranks 0 and 1, got {allreduce}")

    counters = {}
    for rank in (0, 1):
        cp = RUN_DIR / f"counters-rank{rank}.json"
        if not cp.exists():
            raise SystemExit(f"missing counters: {cp}")
        counters[rank] = json.loads(cp.read_text())

    rc = {}
    for rank in (0, 1):
        rp = RUN_DIR / f"rank{rank}.rc"
        if not rp.exists():
            raise SystemExit(f"missing rank rc: {rp}")
        rc[rank] = int(rp.read_text().strip())

    has_ib = all(re.search(r"NET/IB : Using \[0\]mlx5_0", text) for _ in (0, 1))
    no_socket = "Initialized NET plugin Socket" not in text
    checks = {
        "both_ranks_have_allreduce": len(allreduce) == 2,
        "expected_sum_is_3": all(v["expected"] == 3.0 for v in allreduce.values()),
        "value_is_exactly_3": all(v["value"] == 3.0 for v in allreduce.values()),
        "zero_elements_wrong": all(v["bad"] == 0 for v in allreduce.values()),
        "max_abs_err_is_zero": all(v["max_abs_err"] == 0.0 for v in allreduce.values()),
        "both_ranks_exit_zero": rc == {0: 0, 1: 0},
        "both_ranks_posted_sends": all(
            counters[r]["post_send_calls"] > 0 and counters[r]["bytes_written"] > 0
            for r in (0, 1)),
        "both_ranks_posted_recvs": all(counters[r]["post_recv_calls"] > 0 for r in (0, 1)),
        "nccl_named_our_ib_device": has_ib,
        "no_socket_fallback": no_socket,
        "no_p2p_claim_in_this_arm": "NUMERICS p2p" not in text,
    }
    status = job_status()
    started = datetime.fromisoformat(status["started_at"])
    finished = datetime.fromisoformat(status["finished_at"])
    wall = (finished - started).total_seconds()

    metrics = {
        "bead": "gpu-sim-7q3w.5",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "FIRST EXPERIMENT ONLY: 2 genuine NCCL ranks, 1 MiB float32 "
            "allreduce, rank r fills with r+1 so the exact sum is 3.0; exact "
            "comparison (0 elements wrong, max_abs_err 0.0), not a tolerance. "
            "The 4096-byte send/recv arm is present but deferred (NUMERICS_P2P=1); "
            "the first p2p attempt sat in the fabric poll loop, so this artifact "
            "makes no p2p claim."
        ),
        "shape": {
            "ranks": 2,
            "n_float32": allreduce[0]["n"],
            "bytes": allreduce[0]["n"] * 4,
            "fill": "rank r fills every element with r+1",
            "expected_exact_sum": 3.0,
        },
        "transport": {
            "chosen": "NCCL NET/IB",
            "hca": "mlx5_0",
            "socket_fallback": not no_socket,
            "devices": "one aliased real GPU per rank via the landed 7q3w.4 "
                       "vdev shims; ECE verbs return success with zeroed options",
        },
        "result": {
            "allreduce": {str(k): v for k, v in allreduce.items()},
            "counters": {str(k): v for k, v in counters.items()},
            "rank_exit_codes": {str(k): v for k, v in rc.items()},
            "checks": checks,
        },
        "error_budget": {
            "real": "NCCL's own IB transport selected mlx5_0; its real GPU kernels "
                    "computed the allreduce through the 7q3w.4 aliased device; "
                    "both ranks post real WRs and move bytes on the shim rings.",
            "modelled": "The fabric is the libibverbs shim plus MAP_SHARED rings; "
                        "the bytes cross a modelled wire, not a physical NIC.",
            "not_modelled": "No 8-node/8-GPU topology, no 8-way collective, no "
                            "training step, and no p2p result in this first arm.",
        },
        "gate": (
            "none -- the final gate result is cited in the submission comment; "
            "no earlier gate is claimed here"
        ),
        "evidence": {
            "run_sh": "experiments/gpu-sim-7q3w.5/run.sh",
            "capture_py": "experiments/gpu-sim-7q3w.5/capture.py",
            "output_log": "experiments/gpu-sim-7q3w.5/output.log",
            "job_json": "experiments/gpu-sim-7q3w.5/job.json",
        },
    }
    job = {
        "bead": "gpu-sim-7q3w.5",
        "gpu_used": True,
        "job_id": JOB_ID,
        "description": status.get("description"),
        "gpu_index": status.get("gpu_index"),
        "command": status.get("command"),
        "started_at_utc": status["started_at"],
        "finished_at_utc": status["finished_at"],
        "wall_s": round(wall, 1),
        "max_runtime_seconds": status.get("max_runtime_seconds"),
        "state": status.get("state"),
        "exit_code": status.get("exit_code"),
        "job_log": str(JOB_LOG),
        "run_dir": str(RUN_DIR),
        "run_sh": "experiments/gpu-sim-7q3w.5/run.sh",
        "capture_py": "experiments/gpu-sim-7q3w.5/capture.py",
    }
    METRICS.write_text(json.dumps(metrics, indent=2) + "\n")
    JOB.write_text(json.dumps(job, indent=2) + "\n")
    print(json.dumps({"checks": checks, "job_exit": status.get("exit_code")}, indent=2))
    return 0 if all(checks.values()) and status.get("exit_code") == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
