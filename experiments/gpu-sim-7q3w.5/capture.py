#!/usr/bin/env python3
"""Capture gpu-sim-7q3w.5's p2p control (A/B/C/D x NREP) from the real job.

Phases, all run by control_p2p.sh on the GPU queue:
  A acceptance : allreduce + p2p,        pump started at device open -> PASS
  B isolation  : p2p alone,              pump started at device open -> PASS
  C no pump    : p2p alone,              no progress thread          -> WEDGE
  D lazy pump  : p2p alone, FAKEIB_LAZY_PUMP=1 (pre-fix: thread on the first
                 poll_cq)                                            -> WEDGE

C shows the pump is NECESSARY; D shows that STARTING IT AT DEVICE OPEN is the
fix (same binary, same arm, only the start point differs).  Every number below
is parsed from the manifest the control wrote, from each run's own logs and
counter files; a missing or malformed value fails the capture.

--gate names the real gate directory for the submitted tip.  It is a required
argument because a gate field that points at nothing is a false claim.
"""

import argparse
import json
import re
import shutil
import statistics
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
GPQUEUE = Path("/home/gabe/Documents/gpqueue/gpqueue.py")
JOB_LOG_DIR = Path("/home/gabe/.local/state/gpqueue/jobs")

# The shim's tracer and the app share stderr and do not serialise their writes,
# so a trace prefix can land in front of an app line (measured:
# 'FAKEIB[0] NUMERICS p2p rank=0 nbytes=4096 sent=1').  The optional prefix keeps
# the parse exact about CONTENT without pretending the streams are atomic.
PREFIX = r"^(?:FAKEIB\[\d+\] )?"
AR_RE = re.compile(
    PREFIX + r"NUMERICS allreduce rank=(\d+) n=(\d+) expected=([\d.]+) "
    r"value=([\d.]+) max_abs_err=([\d.eE+-]+) bad=(\d+)$", re.M)
P2P_RE = re.compile(PREFIX + r"NUMERICS p2p rank=(\d+) nbytes=(\d+) (.+)$", re.M)
WEDGE_RE = re.compile(r"poll_cq .* -> 0 \(rings out=(\d+) in=(\d+)\)")


def job_status(job_id):
    cp = subprocess.run([sys.executable, str(GPQUEUE), "status", "--json"],
                        capture_output=True, text=True, check=True)
    for job in json.loads(cp.stdout):
        if job["id"] == job_id:
            return job
    raise SystemExit(f"job {job_id} not in gpqueue status")


def read(path):
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"missing evidence file: {p}")
    return p.read_text(errors="replace")


def run_evidence(rec):
    """Facts for ONE run, read from ITS run dir (never from another tip)."""
    run_dir = rec["run_dir"]
    if not run_dir or not Path(run_dir).is_dir():
        raise SystemExit(f"run dir missing for {rec['phase']}{rec['rep']}: {run_dir!r}")
    r0 = read(Path(run_dir) / "rank0.log")
    r1 = read(Path(run_dir) / "rank1.log")
    both = r0 + r1
    allreduce = {}
    for m in AR_RE.finditer(both):
        allreduce[int(m.group(1))] = {
            "n": int(m.group(2)), "expected": float(m.group(3)),
            "value": float(m.group(4)), "max_abs_err": float(m.group(5)),
            "bad": int(m.group(6)),
        }
    p2p = {}
    for m in P2P_RE.finditer(both):
        tail = m.group(3)
        entry = {"nbytes": int(m.group(2))}
        mm = re.search(r"mismatches=(\d+)", tail)
        if mm:
            entry["mismatches"] = int(mm.group(1))
        if "sent=1" in tail:
            entry["sent"] = 1
        p2p[int(m.group(1))] = entry
    counters = {}
    for r in (0, 1):
        cp = Path(run_dir) / f"counters-rank{r}.json"
        # A rank killed by the rank timeout never reaches the destructor; that is
        # expected for C/D and those counters are never read for a wire claim.
        counters[r] = json.loads(cp.read_text()) if cp.exists() else None
    wedges = [(int(a), int(b)) for a, b in WEDGE_RE.findall(r1)]
    return {
        "phase": rec["phase"], "rep": rec["rep"], "rc": rec["rc"],
        "wall_s": rec["wall_s"], "run_dir": run_dir,
        "rank_rc": {"0": rec["rank0_rc"], "1": rec["rank1_rc"]},
        "allreduce": {str(k): v for k, v in allreduce.items()},
        "p2p": {str(k): v for k, v in p2p.items()},
        "counters": {str(k): v for k, v in counters.items()},
        "rank1_posts_conn": bool(rec["rank1_posts_conn"]),
        "rank0_sees_inbound": bool(rec["rank0_sees_inbound"]),
        "rank0_posts_payload": bool(rec["rank0_posts_payload"]),
        "rank1_gets_payload": bool(rec["rank1_gets_payload"]),
        "rank1_p2p_line": bool(rec["rank1_p2p_line"]),
        "wedged": bool(wedges),
        "wedge_last_rings": wedges[-1] if wedges else None,
        "nccl_named_ib_device": "NET/IB : Using [0]mlx5_0" in both,
        "socket_fallback": "Initialized NET plugin Socket" in both,
    }


def exact_ar(ev):
    return (set(ev["allreduce"]) == {"0", "1"} and
            all(v["bad"] == 0 and v["max_abs_err"] == 0.0 and
                v["value"] == v["expected"] for v in ev["allreduce"].values()))


def exact_p2p(ev, nbytes=4096):
    # rank0 reports "sent=1", rank1 reports "mismatches=N": the claim is that the
    # sender posted the payload AND the received bytes equal the sent bytes.
    return (set(ev["p2p"]) == {"0", "1"} and
            ev["p2p"]["0"].get("sent") == 1 and
            ev["p2p"]["1"].get("mismatches") == 0 and
            all(v["nbytes"] == nbytes for v in ev["p2p"].values()))


def clean_wire(ev):
    return all(ev["counters"][r] is not None and
               ev["counters"][r]["rnr_drops"] == 0 and
               ev["counters"][r]["remote_access_errors"] == 0 for r in ("0", "1"))


def wedge_case(ev):
    """The wedge as a POSITIVE statement about both ends, not just an exit code.

    Measured shape (run-20261122T170610 and every C/D run): the RECEIVER
    completed its side -- it posted its 64-byte connection info to the ring
    (out=1) -- and then consumed nothing (in=0).  The SENDER never saw that
    record (no deliver_inbound at all), so it never posted the 4096-byte data
    WR, and the receiver never got the payload.  'Both ranks exited 124' would
    say none of that."""
    return (ev["rc"] != 0 and ev["rank_rc"]["1"] == "124" and
            ev["rank_rc"]["0"] == "124" and
            ev["rank1_posts_conn"] and          # receiver's side reached the wire
            not ev["rank0_sees_inbound"] and    # sender never saw it
            not ev["rank0_posts_payload"] and   # ... so the data WR never posted
            not ev["rank1_gets_payload"] and    # receiver never got the bytes
            not ev["rank1_p2p_line"] and
            ev["wedged"] and ev["wedge_last_rings"] is not None and
            ev["wedge_last_rings"][0] == 1 and  # its own one record (the conn info)
            ev["wedge_last_rings"][1] == 0)     # and it consumed nothing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-id", required=True)
    ap.add_argument("--control-dir", required=True)
    ap.add_argument("--gate", required=True,
                    help="real gate dir for the submitted tip (e.g. "
                         ".lab/gate-logs/<sha>-<ts>)")
    args = ap.parse_args()

    job_log = JOB_LOG_DIR / f"{args.job_id}.log"
    if not job_log.exists():
        raise SystemExit(f"job log missing: {job_log}")
    (HERE / "output.log").write_text(job_log.read_text())

    control_dir = Path(args.control_dir)
    manifest_path = control_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    nrep = manifest["config"]["nrep"]
    wedge_timeout = manifest["config"]["wedge_timeout_s"]
    recs = manifest["records"]
    if len(recs) != nrep * 4:
        raise SystemExit(f"expected {nrep*4} records, got {len(recs)}")

    by_phase = defaultdict(list)
    for rec in recs:
        by_phase[rec["phase"]].append(run_evidence(rec))
    for phase in "ABCD":
        if len(by_phase[phase]) != nrep:
            raise SystemExit(f"phase {phase}: {len(by_phase[phase])} runs, want {nrep}")

    # gate.sh writes .lab/gate-logs/ into the INTEGRATION tree, not into this
    # worktree, so a repo-relative gate path is resolved against REPO.
    gate_dir = Path(args.gate)
    if not gate_dir.is_dir():
        # gate.sh writes .lab/gate-logs/ into the INTEGRATION tree (the common
        # git dir's parent), not into this worktree.
        common = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "--path-format=absolute",
             "--git-common-dir"], capture_output=True, text=True, check=True)
        gate_dir = Path(common.stdout.strip()).parent / args.gate
    if not gate_dir.is_dir():
        raise SystemExit(f"gate dir does not exist: {args.gate} (tried the "
                         f"integration tree too)")

    # Copy the per-run run.sh logs and the manifest next to this script: the
    # control dir lives under .lab/tmp (gitignored, reaped), the artifact must not.
    for rec in recs:
        src = control_dir / f"{rec['phase']}-{rec['rep']}.log"
        if not src.exists():
            raise SystemExit(f"phase log missing: {src}")
        shutil.copyfile(src, HERE / src.name)
    shutil.copyfile(manifest_path, HERE / "manifest.json")

    def walls(phase):
        return [ev["wall_s"] for ev in by_phase[phase]]

    b_walls = walls("B")
    cd_walls = walls("C") + walls("D")
    margin_ratio = min(cd_walls) / max(b_walls) if max(b_walls) > 0 else 0.0

    checks = {
        "A_all_runs_exit_zero": all(ev["rc"] == 0 for ev in by_phase["A"]),
        "A_allreduce_exact_every_run": all(exact_ar(ev) for ev in by_phase["A"]),
        "A_p2p_byte_identical_every_run": all(exact_p2p(ev) for ev in by_phase["A"]),
        "A_wire_clean_every_run": all(clean_wire(ev) for ev in by_phase["A"]),
        "B_all_runs_exit_zero": all(ev["rc"] == 0 for ev in by_phase["B"]),
        "B_p2p_byte_identical_every_run": all(exact_p2p(ev) for ev in by_phase["B"]),
        "B_wire_clean_every_run": all(clean_wire(ev) for ev in by_phase["B"]),
        "B_no_allreduce_claim_any_run": all(not ev["allreduce"] for ev in by_phase["B"]),
        # C: necessity of the progress thread.
        "C_wedges_every_run": all(wedge_case(ev) for ev in by_phase["C"]),
        # D: the MECHANISM -- lazy start (pre-fix) wedges too.
        "D_wedges_every_run": all(wedge_case(ev) for ev in by_phase["D"]),
        # The passing case shows the sender POSTING and the receiver DELIVERING,
        # which is exactly the pair C/D lack (B carries trace for this).
        "B_sender_posted_the_payload_every_run": all(
            ev["rank0_posts_payload"] for ev in by_phase["B"]),
        "B_receiver_delivered_the_payload_every_run": all(
            ev["rank1_gets_payload"] for ev in by_phase["B"]),
        # The margin is a number: B finishes far inside the timeout C/D exhaust.
        "B_wall_under_quarter_of_wedge_timeout":
            max(b_walls) * 4 < wedge_timeout,
        "C_D_wall_at_least_wedge_timeout": min(cd_walls) >= wedge_timeout,
        "much_margin_between_B_and_C_D": margin_ratio >= 4.0,
        "nccl_named_our_ib_device_every_run": all(
            ev["nccl_named_ib_device"] for phase in "ABCD" for ev in by_phase[phase]),
        "no_socket_fallback_any_run": not any(
            ev["socket_fallback"] for phase in "ABCD" for ev in by_phase[phase]),
    }

    def dist(vals):
        return {"min": min(vals), "median": round(statistics.median(vals), 2),
                "max": max(vals), "values": vals}

    phases_out = {}
    for phase in "ABCD":
        evs = by_phase[phase]
        phases_out[phase] = {
            "n": len(evs),
            "rcs": [ev["rc"] for ev in evs],
            "rank_rc": [ev["rank_rc"] for ev in evs],
            "wall_s": [ev["wall_s"] for ev in evs],
            "rank1_posts_conn": [ev["rank1_posts_conn"] for ev in evs],
            "rank0_sees_inbound": [ev["rank0_sees_inbound"] for ev in evs],
            "rank0_posts_payload": [ev["rank0_posts_payload"] for ev in evs],
            "rank1_gets_payload": [ev["rank1_gets_payload"] for ev in evs],
            "rank1_p2p_line": [ev["rank1_p2p_line"] for ev in evs],
            "wedge_last_rings": [ev["wedge_last_rings"] for ev in evs],
            "p2p_mismatches": [ev["p2p"].get("1", {}).get("mismatches")
                               for ev in evs],
            "allreduce_bad": [ev["allreduce"].get("0", {}).get("bad") for ev in evs],
        }

    status = job_status(args.job_id)
    started = datetime.fromisoformat(status["started_at"])
    finished = datetime.fromisoformat(status["finished_at"])

    metrics = {
        "bead": "gpu-sim-7q3w.5",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            f"2 genuine NCCL ranks on this box's one real GPU behind the 7q3w.4 "
            f"aliased-device shims, over the modelled IB wire. A measured BOTH the "
            f"result claims (exact 1 MiB allreduce, rank r fills r+1 => 3.0; and a "
            f"4096-byte send/recv that must be BYTE-IDENTICAL), and the control "
            f"separates two different questions: C (no progress thread) shows the "
            f"async pump is NECESSARY; D (FAKEIB_LAZY_PUMP=1, the pre-fix lazy "
            f"start) shows that STARTING IT AT DEVICE OPEN is the fix. B is the "
            f"p2p claim on its own, with no prior collective. Each phase ran {nrep}x."
        ),
        "shape_chosen": {
            "answer": "neither (a) host compute nor (b) kernel execution",
            "why": (
                "With 7q3w.4's aliasing the kernel launches are REAL, so the "
                "arithmetic is real float32 on this box's GPU; only the TRANSPORT "
                "between the two processes is modelled."
            ),
        },
        "config": {"nrep": nrep, "wedge_timeout_s": wedge_timeout},
        "phases": phases_out,
        "margin": {
            "phase_B_wall_s": dist(b_walls),
            "phase_C_D_wall_s": dist(cd_walls),
            "ratio_min_wedge_wall_over_max_B_wall": round(margin_ratio, 2),
            "meaning": (
                "B completes in well under a quarter of the rank timeout, while "
                "every C and D run was still wedged when the timeout killed it; "
                "the wedge is a deadlock, not a slow path."
            ),
        },
        "wedge_signature": {
            "where": "rank1 (receiver) log of every C and D run",
            "shape": "rank1 (receiver) posts its 64-byte connection info -- "
                     "ring_put(1,0) kind=2 nbytes=64, so its poll line reads "
                     "out=1 in=0 -- and rank0 (sender) has NO deliver_inbound "
                     "line at all: it never saw the record, never posted the "
                     "4096-byte data WR, and rank1 never received a payload. "
                     "The passing runs (B) show the missing pair: rank0 posts "
                     "len=4096 and rank1 delivers nbytes=4096.",
        },
        "result": {"checks": checks},
        "error_budget": {
            "measured_real": (
                "NCCL 2.32.6 chose our mlx5_0 (NET/IB, no Socket fallback in any "
                "run); its own kernels ran on the real GPU; rank0's trace shows the "
                "4096-byte payload posted to the wire; counters show "
                "post_send_calls/bytes_written/ring records > 0 and no rnr_drops or "
                "remote_access_errors in every passing run."
            ),
            "modelled": (
                "The wire: libibverbs is a shim and the fabric is a MAP_SHARED byte "
                "ring per rank pair with modelled bandwidth/latency on completion "
                "readiness. Inbound progress is asynchronous (pump thread, 50 us "
                "default), which adds real delivery latency; that latency is the "
                "pump, not the modelled wire time."
            ),
            "not_modelled": (
                "No physical NIC or DMA, no GDR/dmabuf registration (7q3w.6, NOT "
                "EXERCISED by design), no second GPU and therefore no B300 compute "
                "time (a collective's compute time is this 5090's). No 8-way "
                "collective or training step here, and no performance claim (ae2v)."
            ),
            "model_defect_found_and_fixed": (
                "Inbound records used to be delivered only inside poll_cq, so a "
                "sender spinning on the peer-written host FIFO could never be "
                "progressed; then the progress thread itself was started lazily from "
                "poll_cq, which a p2p-alone sender never calls. Both are measured: "
                "C and D wedge, B does not."
            ),
        },
        "gate": (
            f"scripts/gate.sh --fast at the submitted tip; gate logs {args.gate}. "
            "Only what that directory shows is claimed there."
        ),
        "evidence": {
            "run_sh": "experiments/gpu-sim-7q3w.5/run.sh",
            "control_sh": "experiments/gpu-sim-7q3w.5/control_p2p.sh",
            "capture_py": "experiments/gpu-sim-7q3w.5/capture.py",
            "output_log": "experiments/gpu-sim-7q3w.5/output.log",
            "manifest": "experiments/gpu-sim-7q3w.5/manifest.json",
            "per_run_logs": sorted(p.name for p in HERE.glob("[ABCD]-[0-9]*.log")),
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
        "wall_s": round((finished - started).total_seconds(), 1),
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
