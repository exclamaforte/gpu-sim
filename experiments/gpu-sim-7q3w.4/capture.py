#!/usr/bin/env python3
"""Run each 7q3w.4 increment separately and write the evidence bundle.

Nothing in output.log, metrics.json or job.json is hand-typed: the artifacts
are written from the same subprocess invocations, including each increment's
MemAvailable before/after and exit code.  The three OK lines and each probe's
own `checks` dict are carried verbatim.

Each increment has its own preflight floor, so the capture waits for that floor
before invoking it.  This is why the three runs are separate rather than one
sequential shell: increment 1's peak would otherwise lower MemAvailable below
increment 3's floor before increment 3 starts.  The probes still re-check their
own floor; a refusal (rc=2) is retried after the memory wait, and every attempt
is visible in output.log.
"""
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[1]
OUT = EXP / "output.log"
JOB = EXP / "job.json"
METRICS = EXP / "metrics.json"
RUN_SH = EXP / "run.sh"

# Mirrors the constants in the probes; the probes re-check them, so this is a
# wait hint, not the authority.  (1) 8*1148+1024; (2) 8*600+1024;
# (3) 8*1194+1024.
INCREMENTS = [
    (1, "eight concurrent node processes + RAM preflight", 10208, "CONCURRENT OK"),
    (2, "torch.distributed world_size 8, one rank per node", 5824, "DIST OK"),
    (3, "12.6M transformer + AdamW, 8 DDP ranks, equal checkpoints", 10576, "TRAIN OK"),
]
WAIT_LIMIT_S = 1800
WALL_BOUND_S = 5400
MAX_ATTEMPTS = 3


def mem_available_mib():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    raise RuntimeError("MemAvailable not in /proc/meminfo")


def wait_for(required, label, log):
    deadline = time.monotonic() + WAIT_LIMIT_S
    while True:
        avail = mem_available_mib()
        if avail >= required:
            return avail
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"{label}: MemAvailable {avail} MiB < {required} MiB after "
                f"{WAIT_LIMIT_S}s; no run launched"
            )
        print(
            f"WAIT {label}: MemAvailable {avail} MiB < {required} MiB; "
            "sleeping 30s",
            file=log,
            flush=True,
        )
        time.sleep(30)


def split_sections(text):
    """Map increment number -> text after its last === increment === marker."""
    sections = {}
    current = None
    for line in text.splitlines():
        m = re.match(r"^=== increment (\d+):", line)
        if m:
            current = int(m.group(1))
            sections[current] = []
        elif current is not None:
            sections[current].append(line)
    return {k: "\n".join(v) for k, v in sections.items()}


def first_json(section):
    for idx, ch in enumerate(section):
        if ch != "{":
            continue
        try:
            obj, _ = json.JSONDecoder().raw_decode(section[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "checks" in obj:
            return obj
    return None


def ok_line(section, token):
    for line in section.splitlines():
        if line.startswith(token):
            return line
    return ""


def verdict(result, line):
    return "PASS" if result and line and all(result.get("checks", {}).values()) else "FAIL"


def increment_entry(section, token):
    result = first_json(section)
    line = ok_line(section, token)
    entry = {
        "verdict": verdict(result, line),
        "ok_line": line,
        "checks": result.get("checks") if result else None,
    }
    if result:
        for key in (
            "rss_sum_mib", "mem_available_mib_start", "mem_available_mib_end",
            "preflight_required_mib", "min_child_mem_available_mib",
            "checkpoint_hashes", "checkpoint_agreement",
        ):
            if key in result:
                entry[key] = result[key]
    return entry


def main():
    started = datetime.now(timezone.utc).isoformat()
    cpu_runs = []
    with open(OUT, "w") as log:
        for number, label, required, token in INCREMENTS:
            print(f"=== increment {number}: {label} ===", file=log, flush=True)
            attempts = 0
            before = after = None
            rc = None
            elapsed = 0.0
            while attempts < MAX_ATTEMPTS:
                attempts += 1
                try:
                    before = wait_for(required, label, log)
                except TimeoutError as exc:
                    print(f"TIMEOUT {exc}", file=log, flush=True)
                    rc = 2
                    break
                t0 = time.monotonic()
                rc = subprocess.run(
                    ["bash", str(RUN_SH), str(number)],
                    cwd=str(ROOT),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                ).returncode
                elapsed += time.monotonic() - t0
                after = mem_available_mib()
                print(
                    f"ATTEMPT {number}.{attempts}: rc={rc} "
                    f"MemAvailable {before} -> {after} MiB",
                    file=log,
                    flush=True,
                )
                if rc == 0:
                    break
                if rc != 2:
                    break
                print(
                    f"RETRY {label}: preflight refused (rc=2); waiting for memory",
                    file=log,
                    flush=True,
                )
            cpu_runs.append({
                "command": f"bash experiments/gpu-sim-7q3w.4/run.sh {number}",
                "exit_code": rc,
                "log": "experiments/gpu-sim-7q3w.4/output.log",
                "what": label,
                "wall_s": round(elapsed, 1),
                "attempts": attempts,
                "preflight_required_mib": required,
                "mem_available_mib_before": before,
                "mem_available_mib_after": after,
            })
            print("", file=log, flush=True)

    text = OUT.read_text()
    sections = split_sections(text)
    result = {
        "increment_1_concurrent_nodes": increment_entry(
            sections.get(1, ""), "CONCURRENT OK"
        ),
        "increment_2_torch_distributed": increment_entry(
            sections.get(2, ""), "DIST OK"
        ),
        "increment_3_training": increment_entry(
            sections.get(3, ""), "TRAIN OK"
        ),
    }

    metrics = {
        "bead": "gpu-sim-7q3w.4",
        "captured_at_utc": started,
        "scope": (
            "8 OS processes x 8 virtual devices (A1/D2). Increments 2 and 3 use "
            "world_size 8, one rank per node, per the PI's increment instruction. "
            "A3 (NCCL topology XML/IB choice), A5 (64-way vs 1-way) and A6 "
            "(timing) are NOT claimed here; they are the PI's scope call."
        ),
        "result": result,
        "evidence": {
            "run_sh": "experiments/gpu-sim-7q3w.4/run.sh",
            "capture_py": "experiments/gpu-sim-7q3w.4/capture.py",
            "output_log": "experiments/gpu-sim-7q3w.4/output.log",
            "job_json": "experiments/gpu-sim-7q3w.4/job.json",
        },
    }

    job = {
        "bead": "gpu-sim-7q3w.4",
        "ts_jobs": [],
        "gpu_used": False,
        "cpu_only": True,
        "why_no_gpu": (
            "the model math is CPU float32 + gloo; CUDA is visible only so "
            "torch's NVML/driver identity and AdamW's accelerator health check "
            "work. No GPU kernel or GPU-training claim is made; A1 is identity."
        ),
        "cpu_runs": cpu_runs,
        "wall_bound_s": WALL_BOUND_S,
        "wall_bound_note": (
            "the bound is the whole three-increment capture; the measured "
            "per-increment wall_s is in cpu_runs. It exists so a loaded "
            "two-CPU box cannot turn a slow run into a false failure."
        ),
    }

    JOB.write_text(json.dumps(job, indent=2) + "\n")
    METRICS.write_text(json.dumps(metrics, indent=2) + "\n")

    print(json.dumps({
        "exit_code": max((r["exit_code"] for r in cpu_runs), default=1),
        "wall_s": round(sum(r["wall_s"] for r in cpu_runs), 1),
        "mem_available_now": mem_available_mib(),
        "increments": {k: v["verdict"] for k, v in result.items()},
    }, indent=2))
    return 0 if all(r["exit_code"] == 0 for r in cpu_runs) else 1


if __name__ == "__main__":
    sys.exit(main())
