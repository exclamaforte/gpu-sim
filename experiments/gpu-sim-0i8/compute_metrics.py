#!/usr/bin/env python3
import json, re, pathlib, sys, argparse

ROOT = pathlib.Path(__file__).resolve().parent
REPO = ROOT.parent.parent
TRACES = REPO / "experiments" / "gpu-sim-pkb" / "traces"
OUT = ROOT / "metrics.json"

def verdict_of(rank_text):
    if "VALUE OK" in rank_text:
        return "VALUE OK"
    m = re.search(r"VALUE MISMATCH:[^\n]*", rank_text)
    if m:
        return m.group(0).strip()
    m = re.search(r"rank \d+ init failed:[^\n]*", rank_text)
    if m:
        return m.group(0).strip()
    m = re.search(r"rank \d+ allreduce failed:[^\n]*", rank_text)
    if m:
        return m.group(0).strip()
    if "FAIL: no 'via NET/sim'" in rank_text:
        return "FAIL (fabric bypassed)"
    return "UNKNOWN"

def parse_run(runid):
    combined = TRACES / f"trace_{runid}.log"
    r0 = TRACES / f"run_{runid}_rank0.txt"
    r1 = TRACES / f"run_{runid}_rank1.txt"
    combined_text = combined.read_text(errors="replace") if combined.exists() else ""
    r0_text = r0.read_text(errors="replace") if r0.exists() else ""
    r1_text = r1.read_text(errors="replace") if r1.exists() else ""

    # size/proto from runid: sd_<size>_<proto>_r<n> or nu_<size>_<proto>_r<n>
    m = re.match(r"(sd|nu)_(\d+)_(\w+)_r(\d+)", runid)
    instrument = "selfdesc" if runid.startswith("sd_") else "nonuniform"
    size = int(m.group(2)) if m else None
    proto = m.group(3) if m else None

    isends = combined_text.count("v12.isend isend")
    via_net = "via NET/sim" in combined_text or "via NET/sim" in r0_text
    if (via_net and isends > 0 and "VALUE OK" in r0_text and "VALUE OK" in r1_text
            and "allreduce failed" not in r0_text and "allreduce failed" not in r1_text
            and "init failed" not in r0_text and "init failed" not in r1_text):
        script_exit = 0
    else:
        script_exit = 1

    return {
        "size": size,
        "protocol": proto,
        "run": int(m.group(4)) if m else None,
        "run_id": runid,
        "script_exit": script_exit,
        "via_net": bool(via_net),
        "isend_lines": isends,
        "rank0_verdict": verdict_of(r0_text),
        "rank1_verdict": verdict_of(r1_text),
        "instrument": instrument,
        "trace_path": str(combined.relative_to(REPO)),
    }

def main():
    global TRACES, OUT, REPO
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default=str(TRACES))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    TRACES = pathlib.Path(args.traces)
    OUT = pathlib.Path(args.out)
    REPO = TRACES.parent.parent
    runids = []
    for p in sorted(TRACES.glob("trace_*.log")):
        rid = p.stem[len("trace_"):]
        if rid.startswith("sd_") or rid.startswith("nu_"):
            runids.append(rid)

    runs = [parse_run(r) for r in runids]

    # cell key
    def cell_key(r):
        return (r["size"], r["protocol"], r["instrument"])

    # qualifying = script_exit 0 and via_net true
    for r in runs:
        r["qualifying"] = (r["script_exit"] == 0 and r["via_net"] is True)

    n_qualifying = sum(1 for r in runs if r["qualifying"])

    # determine failing cells
    cells = {}
    for r in runs:
        cells.setdefault(cell_key(r), []).append(r)

    failing_cells = []
    for ck, rs in cells.items():
        # a cell fails if not all runs qualifying+passing
        size, proto, instr = ck
        cell_pass = True
        if len(rs) < 3:
            cell_pass = False
        for r in rs:
            if not r["qualifying"]:
                cell_pass = False
                break
            if instr == "selfdesc":
                if r["rank0_verdict"] != "VALUE OK" or r["rank1_verdict"] != "VALUE OK":
                    cell_pass = False
                    break
            else:
                if r["rank0_verdict"] != "VALUE OK" or r["rank1_verdict"] != "VALUE OK":
                    cell_pass = False
                    break
        if not cell_pass:
            failing_cells.append(f"{size}:{proto}:{instr}")

    failing_cells = sorted(set(failing_cells))

    # decision
    if not failing_cells and len(runs) > 0:
        decision = "CERTIFIED"
    elif any(r["qualifying"] and ("MISMATCH" in r["rank0_verdict"] or "MISMATCH" in r["rank1_verdict"]) for r in runs):
        decision = "DEFECT-STANDS"
    else:
        decision = "PARTIAL"

    # remove helper key before writing
    for r in runs:
        r.pop("qualifying", None)

    metrics = {
        "runs": runs,
        "n_qualifying_runs": n_qualifying,
        "decision": decision,
        "failing_cells": failing_cells,
        "pre_registered_at": "0fa2e868-e7c5-502d-9dd0-24decfb642bc",
    }

    OUT.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"wrote {OUT} with {len(runs)} runs, {n_qualifying} qualifying, decision={decision}")

if __name__ == "__main__":
    main()
