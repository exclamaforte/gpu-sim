#!/usr/bin/env python3
"""gpu-sim-8g6 verdict builder (CPU): assemble metrics.json from the three
committed probe-job records. Deterministic; run.sh re-runs it to scratch
and byte-compares. Exits nonzero if any expectation breaks (e.g. a marker
file shows a member call -- that would CONTRADICT the recorded verdict and
must fail loudly, not silently pass).

Usage: python3 experiments/gpu-sim-8g6/build_verdict.py --metrics-out <path>
"""
import argparse
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

JOBS = {
    "job1": "20261006t053031-9ef7a761",
    "job2": "20261006t053152-7b064477",
    "job3": "20261006t053533-36d82f0a",
}


def read_markers(tag):
    rows = []
    for line in (HERE / tag / "markers.jsonl").read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-out", required=True)
    args = ap.parse_args()
    fails = []
    jobs = {}
    for tag, jid in JOBS.items():
        markers = read_markers(tag)
        log = (HERE / tag / "nccl_debug.log").read_text()
        members = [m["member"] for m in markers
                   if m["member"] not in ("LOADED",) and not m["member"].startswith("ENTRY_")]
        entries = [m["member"] for m in markers if m["member"].startswith("ENTRY_")]
        jobs[tag] = {
            "ts_job": jid,
            "markers": markers,
            "member_calls": members,
            "entry_calls": entries,
            "nccl_log_lines": len(log.splitlines()),
            "plugin_path_logged": "NCCL_NET_PLUGIN set by environment" in log,
        }
    # The verdict's load-bearing facts, re-asserted as checks:
    if jobs["job1"]["markers"] != []:
        fails.append("job1: expected empty markers (pre-stdbuf run)")
    if jobs["job2"]["markers"] != [{"member": "LOADED", "off": -1}]:
        fails.append("job2: expected exactly [LOADED]")
    if jobs["job3"]["markers"] != [{"member": "LOADED", "off": -1}]:
        fails.append("job3: expected exactly [LOADED] (no ENTRY)")
    for tag in JOBS:
        if jobs[tag]["member_calls"]:
            fails.append(f"{tag}: a member stub FIRED -- verdict contradicted")
        if tag != "job1" and not jobs[tag]["plugin_path_logged"]:
            fails.append(f"{tag}: NCCL log does not show the plugin path")
    # 21 distinct stubs + 6 entries in the committed probe source.
    src = (HERE / "probe_v11_markers.c").read_text()
    import re
    stubs = sorted(set(re.findall(r"stub_(\w+)", src)))
    entries = sorted(set(re.findall(r"ncclNetPlugin(_v\d+)", src)))
    if len(stubs) != 21:
        fails.append(f"probe source has {len(stubs)} distinct stubs, want 21")
    if entries != ["_v10", "_v11", "_v6", "_v7", "_v8", "_v9"]:
        fails.append(f"probe source entries {entries} != v6..v11")

    def sh(*a):
        try:
            return subprocess.run(["git"] + list(a), capture_output=True,
                                    text=True, cwd=str(REPO),
                                    timeout=30).stdout.strip()
        except Exception:
            return "unknown"
    metrics = {
        "bead": "gpu-sim-8g6",
        "tree": {"branch": sh("branch", "--show-current"),
                 "commit": sh("rev-parse", "--short", "HEAD")},
        "ts_jobs": [JOBS[t] for t in ("job1", "job2", "job3")],
        "jobs": jobs,
        "probe_source": {"members": len(stubs), "entries": entries,
                         "layout_bytes": 176},
        "fault": {"signal": "SIGSEGV", "exit": 139,
                   "address_captured": False,
                   "note": "no core captured; localization is by marker "
                           "absence (LOADED without ENTRY without member), "
                           "not by fault address"},
        "member_mapping": {"reached_member": None,
                           "explanation": "no member stub fired in any of 3 "
                                          "jobs; stub_symbols.txt in each job "
                                          "dir maps the 21 distinct symbols "
                                          "had one fired"},
        "verdict": "NO MEMBER REACHED -- fault at entry invocation or "
                   "earlier (dlopen ok, NCCL_NET_PLUGIN read, "
                   "ncclNetPlugin_v11 never entered). Member-level "
                   "calling-convention recovery is BLOCKED: the loader never "
                   "gets far enough for member signatures to matter.",
        "followup": "LD_DEBUG=symbols run would show whether "
                    "ncclNetPlugin_v11 is even dlsym'd (vs fault before "
                    "resolution); named, not run -- three GPU jobs answer "
                    "this bead, a fourth needs its own bead.",
        "installed_untouched": True,
        "verdict_final": "pass" if not fails else "fail",
        "failures": fails,
    }
    Path(args.metrics_out).write_text(json.dumps(metrics, indent=2,
                                                 sort_keys=True) + "\n")
    print(f"member_calls_total=0 verdict={metrics['verdict_final']} -> {args.metrics_out}",
          flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
