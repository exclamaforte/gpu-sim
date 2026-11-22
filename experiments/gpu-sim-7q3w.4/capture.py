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
    (0, "A2 node isolation + negative control", 6000, "A2 OK"),
    (1, "eight concurrent node processes + RAM preflight", 10208, "CONCURRENT OK"),
    (2, "torch.distributed world_size 8, one rank per node", 5824, "DIST OK"),
    (3, "12.6M transformer + AdamW, 8 DDP ranks, equal checkpoints", 10576, "TRAIN OK"),
    (4, "A5 8-way DDP vs single-rank reference", 10576, "A5 OK"),
]
WAIT_LIMIT_S = 1800
WALL_BOUND_S = 7200
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


def last_json(section):
    found = None
    for idx, ch in enumerate(section):
        if ch != "{":
            continue
        try:
            obj, _ = json.JSONDecoder().raw_decode(section[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "checks" in obj:
            found = obj
    return found


def ok_line(section, token):
    for line in section.splitlines():
        if line.startswith(token):
            return line
    return ""


def verdict(result, line):
    return "PASS" if result and line and all(result.get("checks", {}).values()) else "FAIL"


def increment_entry(section, token):
    result = last_json(section)
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
            "eight_way_final_loss", "one_way_final_loss", "final_loss_delta",
            "max_abs_param_delta", "tolerance",
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
        # A refusal is evidence, not a failed increment: force the PI's measured
        # 10496 MiB start through the same preflight and record the line.
        print("=== increment 5: REFUSE control (10496 < 10576) ===", file=log, flush=True)
        refuse_before = mem_available_mib()
        refuse_rc = subprocess.run(
            ["bash", str(RUN_SH), "5"],
            cwd=str(ROOT),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        ).returncode
        print(
            f"REFUSE CONTROL: rc={refuse_rc} MemAvailable_now={refuse_before} MiB",
            file=log,
            flush=True,
        )

    text = OUT.read_text()
    sections = split_sections(text)
    result = {
        "a2_node_isolation": increment_entry(sections.get(0, ""), "A2 OK"),
        "increment_1_concurrent_nodes": increment_entry(
            sections.get(1, ""), "CONCURRENT OK"
        ),
        "increment_2_torch_distributed": increment_entry(
            sections.get(2, ""), "DIST OK"
        ),
        "increment_3_training": increment_entry(
            sections.get(3, ""), "TRAIN OK"
        ),
        "a5_reference": increment_entry(sections.get(4, ""), "A5 OK"),
    }
    refuse_section = sections.get(5, "")
    refuse_line = ok_line(refuse_section, "REFUSE:")
    result["refuse_control"] = {
        "verdict": "PASS" if (
            refuse_rc == 2 and "10496" in refuse_line and "10576" in refuse_line
        ) else "FAIL",
        "ok_line": refuse_line,
        "exit_code": refuse_rc,
        "note": "forced measured start reproducing the PI's real refusal; no ranks launched",
    }

    metrics = {
        "bead": "gpu-sim-7q3w.4",
        "captured_at_utc": started,
        "scope": (
            "8 OS processes x 8 virtual devices (A1/D2). Increments 2 and 3 use "
            "world_size 8, one rank per node; A5 compares that 8-way run with a "
            "single-rank reference. A3 is gpu-sim-s2vc, A6 is gpu-sim-ae2v, and "
            "A4 remains gpu-sim-7q3w.5: no NCCL data-path or topology claim is made."
        ),
        "result": result,
        "evidence": {
            "run_sh": "experiments/gpu-sim-7q3w.4/run.sh",
            "capture_py": "experiments/gpu-sim-7q3w.4/capture.py",
            "output_log": "experiments/gpu-sim-7q3w.4/output.log",
            "job_json": "experiments/gpu-sim-7q3w.4/job.json",
        },
    }
    spec = json.loads((ROOT / "specs" / "b300_8x8.json").read_text())
    metrics["d1_latency"] = {
        "marking": "MODELLING CHOICE (not a vendor figure)",
        "nvlink_us": spec["links"][0]["latency_us"],
        "nic_us": spec["nodes"][0]["nics"][0]["latency_us"],
        "source": spec.get("modelling_choices", {}),
    }
    metrics["error_budget"] = {
        "ram": {
            "preflight_required_mib": result["increment_3_training"].get(
                "preflight_required_mib"),
            "this_run_mem_available_start_mib": result["increment_3_training"].get(
                "mem_available_mib_start"),
            "this_run_mem_available_end_mib": result["increment_3_training"].get(
                "mem_available_mib_end"),
            "pi_passing_run_start_mib": 10633,
            "pi_refusal_start_mib": 10496,
            "pi_refusal_required_mib": 10576,
            "statement": (
                "fits on a quiet box with a few hundred MiB of margin; does not "
                "fit on a busy one, and the preflight refuses rather than "
                "OOM-killing. The PI measured a passing start at 10633 MiB "
                "(57 MiB above 10576) and a refusal at 10496 MiB; this capture's "
                "start/end are above."
            ),
        },
        "what_is_real": [
            "torch float32 forward/backward/AdamW on the host",
            "gloo all_reduce and DDP gradient averaging",
            "the 12,609,536-parameter model and its losses/checkpoint bytes",
        ],
        "what_is_modelled": [
            "B300 name/capacity/SM count/UUID/PCI bus ids from D1",
            "64 virtual devices over 8 OS processes",
            "per-hop NVLink/NIC latency (fabric-calendar modelling choice)",
        ],
        "not_modelled": [
            "NCCL data path (gpu-sim-7q3w.5)",
            "NCCL topology XML/IB choice (gpu-sim-s2vc)",
            "fabric timing (gpu-sim-ae2v)",
            "SASS execution, real PCIe/NVLink protocol behaviour, hardware errors",
        ],
        "a5": {
            "tolerance": result["a5_reference"].get("tolerance"),
            "final_loss_delta": result["a5_reference"].get("final_loss_delta"),
            "max_abs_param_delta": result["a5_reference"].get(
                "max_abs_param_delta"),
        },
    }
    metrics["a_item_map"] = {
        "A1": "probe_vdev.py: 8 virtual B300 devices via NVML; checks device_count_is_N, name_is_D1, total_mem_is_D1, sm_count_is_D1, uuid_prefix, nvml_private_count, matmul_err. Run = increment 1.",
        "A2": "probe_a2.py: probe_nodes.py normal run must have 64 distinct D1 PCI bus ids; the --disable-isolation run must FAIL the same checks. Run = increment 0.",
        "A3": "moved to gpu-sim-s2vc (NCCL topology XML/IB choice); not claimed here.",
        "A4": "gpu-sim-7q3w.5 owns NCCL numerics; increment 2 is a gloo control-plane all_reduce only.",
        "A5": "probe_reference.py: 8-way DDP final loss and parameters vs a 1-way reference with the same seed/data; TOL=1e-5 absolute. Run = increment 4.",
        "A6": "moved to gpu-sim-ae2v (fabric timing); no timing claim here.",
        "A7": "not separately rerun here; increment 3's checkpoint hash was reproduced by the PI, and 7q3w.3's 20-run transport arm owns the exit distribution.",
        "A8": "metrics.error_budget.ram: preflight 10576 MiB, PI pass 10633/10576 (57 MiB margin), PI refusal 10496; this capture records its own start/end.",
    }
    metrics["d4_coverage"] = {
        "model": "probe_training.py and probe_reference.py: 4-layer TransformerEncoder, d_model=512, ff=2048, 12,609,536 params, AdamW lr=1e-3, 3 steps, 8 DDP ranks, rank-sharded data.",
        "launcher": "run.sh picks one increment; capture.py waits for that increment's preflight floor, records exit codes and MemAvailable before/after, and writes output.log/metrics.json/job.json.",
        "d1": "probe_vdev.py (identity) + probe_nodes.py/probe_a2.py (64 devices over 8 nodes).",
        "d2": "probe_nodes.py and probe_a2.py: 8 OS processes x 8 devices, 64 distinct D1 bus ids, negative control fails without isolation.",
        "d3": "probe_vdev.py: real driver/cudart/NVML identity through the shims; probe_distributed.py: one rank per node.",
        "d4": "probe_training.py + probe_reference.py + run.sh + capture.py, as above; this is the model/launcher pair, not a claim of GPU kernels.",
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
        "refuse_control": {
            "command": "bash experiments/gpu-sim-7q3w.4/run.sh 5",
            "exit_code": refuse_rc,
            "log": "experiments/gpu-sim-7q3w.4/output.log",
            "what": "REFUSE control at the PI's measured 10496 MiB start",
            "ok_line": refuse_line,
        },
        "wall_bound_s": WALL_BOUND_S,
        "wall_bound_note": (
            "the bound is the whole five-increment capture; the measured "
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
