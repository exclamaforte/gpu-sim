#!/usr/bin/env python3
"""Capture gpu-sim-7q3w.5's p2p acceptance run + its negative control.

The job ran on the real GPU queue (control_p2p.sh): phase A is the acceptance
shape (allreduce + p2p with the asynchronous inbound pump), phase B isolates the
p2p arm (no earlier collective), phase C is the NEGATIVE control (same p2p arm
with FAKEIB_NO_PUMP_THREAD=1, which must wedge).  Every value below is parsed
from the job log, the phase run dirs' counter files and rc files, or the
manifest the control wrote; a missing or malformed value fails the capture
instead of being defaulted.
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
GPQUEUE = Path("/home/gabe/Documents/gpqueue/gpqueue.py")
JOB_LOG_DIR = Path("/home/gabe/.local/state/gpqueue/jobs")

AR_RE = re.compile(
    r"^NUMERICS allreduce rank=(\d+) n=(\d+) expected=([\d.]+) "
    r"value=([\d.]+) max_abs_err=([\d.eE+-]+) bad=(\d+)$", re.M)
P2P_RE = re.compile(r"^NUMERICS p2p rank=(\d+) nbytes=(\d+) (.*)$", re.M)


def job_status(job_id):
    cp = subprocess.run([sys.executable, str(GPQUEUE), "status", "--json"],
                        capture_output=True, text=True, check=True)
    for job in json.loads(cp.stdout):
        if job["id"] == job_id:
            return job
    raise SystemExit(f"job {job_id} not in gpqueue status")


def counters(run_dir, rank):
    path = Path(run_dir) / f"counters-rank{rank}.json"
    if not path.exists():
        raise SystemExit(f"missing counters: {path}")
    return json.loads(path.read_text())


def rank_rc(run_dir, rank):
    path = Path(run_dir) / f"rank{rank}.rc"
    if not path.exists():
        raise SystemExit(f"missing rank rc: {path}")
    return int(path.read_text().strip())


def phase_evidence(name, phase, parse_p2p=True):
    """One phase's measured facts, read from ITS run dir and ITS log only."""
    run_dir = phase["run_dir"]
    if not run_dir or not Path(run_dir).is_dir():
        raise SystemExit(f"phase {name}: run dir missing: {run_dir!r}")
    text = (Path(run_dir) / "rank0.log").read_text(errors="replace")
    text += (Path(run_dir) / "rank1.log").read_text(errors="replace")
    allreduce = {}
    for m in AR_RE.finditer(text):
        allreduce[int(m.group(1))] = {
            "n": int(m.group(2)),
            "expected": float(m.group(3)),
            "value": float(m.group(4)),
            "max_abs_err": float(m.group(5)),
            "bad": int(m.group(6)),
        }
    p2p = {}
    if parse_p2p:
        for m in P2P_RE.finditer(text):
            tail = m.group(3)
            entry = {"nbytes": int(m.group(2)), "raw": tail}
            mm = re.search(r"mismatches=(\d+)", tail)
            if mm:
                entry["mismatches"] = int(mm.group(1))
            p2p[int(m.group(1))] = entry
    return {
        "rc": phase["rc"],
        "run_dir": run_dir,
        "log": phase["log"],
        "rank_rc": {str(r): rank_rc(run_dir, r) for r in (0, 1)},
        "counters": {str(r): counters(run_dir, r) for r in (0, 1)},
        "allreduce": {str(k): v for k, v in allreduce.items()},
        "p2p": {str(k): v for k, v in p2p.items()},
        "nccl_named_ib_device": "NET/IB : Using [0]mlx5_0" in text,
        "socket_fallback": "Initialized NET plugin Socket" in text,
        "wedge_signature": "poll_cq" in text and re.search(
            r"poll_cq .* -> 0 \(rings out=", text) is not None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-id", default="20261123t002523-5c988721")
    ap.add_argument("--control-dir", default=None,
                    help="control-<ts> dir; resolved from the job log when omitted")
    args = ap.parse_args()

    job_log = JOB_LOG_DIR / f"{args.job_id}.log"
    if not job_log.exists():
        raise SystemExit(f"job log missing: {job_log}")
    text = job_log.read_text()
    (HERE / "output.log").write_text(text)

    control_dir = args.control_dir
    if control_dir is None:
        m = re.search(r"^--- control dir (.+)$", text, re.M)
        if not m:
            raise SystemExit("job log has no 'control dir' line")
        control_dir = m.group(1).strip()
    control_dir = Path(control_dir)
    manifest_path = control_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    phases = manifest["phases"]

    A = phase_evidence("A", phases["A"])
    B = phase_evidence("B", phases["B"])
    C = phase_evidence("C", phases["C"])

    # Copy the phase logs + manifest next to this script: the control dir lives
    # under .lab/tmp (gitignored, reaped), the artifact must not.
    for phase in phases.values():
        src = control_dir / phase["log"]
        if not src.exists():
            raise SystemExit(f"phase log missing: {src}")
        shutil.copyfile(src, HERE / phase["log"])
    shutil.copyfile(manifest_path, HERE / "manifest.json")

    def clean_wire(ph):
        return all(ph["counters"][r]["rnr_drops"] == 0 and
                   ph["counters"][r]["remote_access_errors"] == 0 for r in ("0", "1"))

    def exact_ar(ph):
        return (set(ph["allreduce"]) == {"0", "1"} and
                all(v["bad"] == 0 and v["max_abs_err"] == 0.0 and
                    v["value"] == v["expected"] for v in ph["allreduce"].values()))

    def exact_p2p(ph, nbytes=4096):
        return (set(ph["p2p"]) == {"0", "1"} and
                all(v["nbytes"] == nbytes for v in ph["p2p"].values()) and
                ph["p2p"]["1"].get("mismatches") == 0)

    checks = {
        "A_ranks_exit_zero": A["rank_rc"] == {"0": 0, "1": 0},
        "A_allreduce_is_exact_on_both_ranks": exact_ar(A),
        "A_p2p_payload_is_byte_identical": exact_p2p(A),
        "A_no_rnr_drop_or_remote_access_error": clean_wire(A),
        "A_nccl_named_our_ib_device": A["nccl_named_ib_device"],
        "A_no_socket_fallback": not A["socket_fallback"],
        "B_p2p_alone_is_byte_identical": B["rc"] == 0 and exact_p2p(B),
        "B_both_ranks_exit_zero": B["rank_rc"] == {"0": 0, "1": 0},
        "B_has_no_allreduce_claim": not B["allreduce"],
        "B_no_rnr_drop_or_remote_access_error": clean_wire(B),
        # The control must fail, and fail for the documented reason.
        "C_without_pump_fails": C["rc"] != 0,
        "C_rank1_hit_the_rank_timeout": C["rank_rc"]["1"] == 124,
        "C_rank1_received_nothing": "1" not in C["p2p"],
        "C_rank1_wedged_in_poll_loop": C["wedge_signature"],
    }

    status = job_status(args.job_id)
    started = datetime.fromisoformat(status["started_at"])
    finished = datetime.fromisoformat(status["finished_at"])
    wall = (finished - started).total_seconds()

    metrics = {
        "bead": "gpu-sim-7q3w.5",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "2 genuine NCCL ranks on this box's one real GPU behind the 7q3w.4 "
            "aliased-device shims, over the modelled IB wire (libibverbs shim + "
            "MAP_SHARED rings). Acceptance shape A measures BOTH: an allreduce of "
            "a 1 MiB float32 vector where rank r fills r+1 (exact sum 3.0) must be "
            "EXACT -- 0 elements wrong, max_abs_err 0.0, not a tolerance -- and a "
            "4096-byte ncclSend/ncclRecv whose received bytes must be "
            "BYTE-IDENTICAL (mismatches=0). Phase B re-runs the p2p arm with no "
            "prior collective. Phase C is the negative control: the same p2p arm "
            "with the asynchronous inbound pump disabled must wedge, which is how "
            "the 7q3w.3 shape failed."
        ),
        "shape_chosen": {
            "answer": "neither (a) host compute nor (b) kernel execution",
            "why": (
                "The 7q3w.4 aliasing shims present virtual B300s that alias the one "
                "real GPU and forward everything they do not override to the real "
                "driver, so KERNEL LAUNCHES ARE REAL: NCCL's allreduce and p2p "
                "kernels execute on the real GPU and the arithmetic is real float32. "
                "Only the transport between the two processes is modelled."
            ),
        },
        "phases": {"A": A, "B": B, "C": C},
        "result": {
            "A_allreduce": A["allreduce"],
            "A_p2p": A["p2p"],
            "C_control_rank_rc": C["rank_rc"],
            "checks": checks,
        },
        "error_budget": {
            "measured_real": (
                "NCCL 2.32.6 chose our mlx5_0 (NET/IB, no Socket fallback); its "
                "own kernels ran on the real GPU; both ranks posted real verbs WRs "
                "and the payload bytes crossed the modelled wire (counters: "
                "post_send_calls > 0, bytes_written > 0, ring records in/out > 0)."
            ),
            "modelled": (
                "The wire: libibverbs is a shim and the 'fabric' is a MAP_SHARED "
                "byte ring per rank pair, with modelled bandwidth/latency applied "
                "to completion readiness. The GPU identity is one real device seen "
                "through N aliases. Delivery is asynchronous (pump thread); the "
                "pump interval (default 50 us) is added real latency, not modelled "
                "wire time."
            ),
            "not_modelled": (
                "No physical NIC, no DMA/PCIe transfer, no GDR/dmabuf registration "
                "(7q3w.6), no second GPU, and therefore no B300 compute time: a "
                "collective's COMPUTE time is this 5090's, while its TRANSPORT time "
                "is the D1 model. No 8-way collective and no training step here; "
                "no performance claim (that is gpu-sim-ae2v)."
            ),
            "model_defect_found_and_fixed": (
                "Inbound records used to be delivered only inside poll_cq, so a "
                "sender spinning on the peer-written host FIFO could never be "
                "progressed (rank1 wedged with rings out=49 in=48). The pump thread "
                "models the NIC writing memory on its own; phase C is that control."
            ),
        },
        "gate": (
            "none -- the final gate result is cited in the submission comment; "
            "no earlier gate is claimed here"
        ),
        "evidence": {
            "run_sh": "experiments/gpu-sim-7q3w.5/run.sh",
            "control_sh": "experiments/gpu-sim-7q3w.5/control_p2p.sh",
            "capture_py": "experiments/gpu-sim-7q3w.5/capture.py",
            "output_log": "experiments/gpu-sim-7q3w.5/output.log",
            "phase_logs": [p["log"] for p in phases.values()],
            "manifest": "experiments/gpu-sim-7q3w.5/manifest.json",
            "job_json": "experiments/gpu-sim-7q3w.5/job.json",
        },
    }

    job = {
        "bead": "gpu-sim-7q3w.5",
        "gpu_used": True,
        "job_id": args.job_id,
        "description": status.get("description"),
        "gpu_index": status.get("gpu_index"),
        "command": status.get("command"),
        "started_at_utc": status["started_at"],
        "finished_at_utc": status["finished_at"],
        "wall_s": round(wall, 1),
        "max_runtime_seconds": status.get("max_runtime_seconds"),
        "state": status.get("state"),
        "exit_code": status.get("exit_code"),
        "job_log": str(job_log),
        "control_dir": str(control_dir),
        "run_sh": "experiments/gpu-sim-7q3w.5/run.sh",
        "capture_py": "experiments/gpu-sim-7q3w.5/capture.py",
    }

    (HERE / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (HERE / "job.json").write_text(json.dumps(job, indent=2) + "\n")
    print(json.dumps({"checks": checks, "job_exit": status.get("exit_code")},
                     indent=2))
    return 0 if all(checks.values()) and status.get("exit_code") == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
