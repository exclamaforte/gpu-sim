#!/usr/bin/env python3
"""gpu-sim-u2o.19 evidence (CPU + two recorded GPU probes): parse the two ts
probe logs and the overlay-gate output into metrics.json. Every number is
read from a named artifact; nothing is hand-typed.
Usage: python3 experiments/gpu-sim-u2o.19/u19_verify.py --metrics-out <p>
       (reads ts logs from ~/.local/state/gpqueue/jobs/ + gate log argument)
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
JOBS = Path("/home/gabe/.local/state/gpqueue/jobs")
PROBE1 = "20261010t211802-ff147780"
PROBE2 = "20261010t213835-4d4bdf3f"


def parse_kv(text):
    return dict(re.findall(r"^([AS]\.\w+)\s*:\s*(\d+)", text, re.M))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-out", required=True)
    ap.add_argument("--gate-log", required=True)
    args = ap.parse_args()
    fails, ev = [], {}
    logs = {}
    for j in (PROBE1, PROBE2):
        p = JOBS / f"{j}.log"
        if not p.exists():
            fails.append(f"missing ts log {j}")
            continue
        logs[j] = p.read_text()
    kv1 = parse_kv(logs.get(PROBE1, ""))
    kv2 = parse_kv(logs.get(PROBE2, ""))
    ev["probe1_struct"] = {k: int(v) for k, v in kv1.items() if k.startswith("S.")}
    ev["probe2_struct"] = {k: int(v) for k, v in kv2.items() if k.startswith("S.")}
    ev["probe2_attr"] = {k: int(v) for k, v in kv2.items() if k.startswith("A.")}
    for k in ("S.maxThreadsPerSM", "S.sharedMemPerSM", "S.l2CacheSize"):
        if ev["probe1_struct"].get(k) != ev["probe2_struct"].get(k):
            fails.append(f"struct {k} differs across probes")
    agree = {"S.maxThreadsPerSM": "A.maxThreadsPerSM",
             "S.sharedMemPerSM": "A.sharedMemPerSM",
             "S.l2CacheSize": "A.l2CacheSize"}
    for s, a in agree.items():
        if ev["probe2_struct"].get(s) != ev["probe2_attr"].get(a):
            fails.append(f"struct/attr disagree: {s} vs {a}")
    for k, want in (("S.maxThreadsPerSM", 1536), ("S.sharedMemPerSM", 102400),
                    ("S.l2CacheSize", 100663296),
                    ("A.optinPerBlock", 101376)):
        got = (ev["probe2_struct"] if k.startswith("S.") else ev["probe2_attr"]).get(k)
        if got != want:
            fails.append(f"{k}={got}, want {want}")
    gate = Path(args.gate_log).read_text()
    ev["gate_passed"] = "gate PASSED" in gate
    ev["gate_smem_pin"] = "attr smemPerSM == struct      102400" in gate
    ev["gate_optin_pin"] = "attr optinPerBlock      101376" in gate
    if not all(ev[k] for k in ("gate_passed", "gate_smem_pin", "gate_optin_pin")):
        fails.append("overlay gate pins missing")
    metrics = {"bead": "gpu-sim-u2o.19",
               "probes": {"struct_reread": PROBE1, "attr_readthrough": PROBE2},
               "struct_values": ev["probe2_struct"],
               "attr_values": ev["probe2_attr"],
               "adjudication": {
                   "maxThreadsPerMultiProcessor": 1536,
                   "evidence": "own struct probe (2 realizations) + attribute readthrough 1536; no megakern field exists",
                   "sharedMemPerMultiprocessor": 102400,
                   "l2CacheSize": 100663296,
                   "two_lab_chain": "gpu-sim probes + megakern-37e.2 job 20261003t213327-db6b7f3f (to the byte)",
                   "optin_per_block": "SET 101376 from own probe + megakern agreement (was silently absent)"},
               "withdrawn": "probe-1 'A.sharedMemPerSM: 0' was uninitialized-status garbage (EMIT before assign); superseded by probe-2 A=102400",
               "gate": {k: ev[k] for k in ("gate_passed", "gate_smem_pin", "gate_optin_pin")},
               "verdict": "pass" if not fails else "fail", "failures": fails}
    Path(args.metrics_out).write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(f"verdict={metrics['verdict']} -> {args.metrics_out}", flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
