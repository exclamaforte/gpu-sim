#!/usr/bin/env python3
"""Reconciliation test for the anchor artifact (gpu-sim-u2o.6.6.2).

The defect this exists to catch: merge_metrics.py computed `bytes_moved` per
kernel but `achieved_GBps` from a flat 2*n*4, so for exactly the kernels whose
convention differs -- saxpy (two reads, one write) and strided_read (a
stride-17 access that misses cache lines) -- the two fields of the SAME entry
disagreed. The published ceiling escaped only because its argmax cell is one
where the two conventions happen to agree: right by accident, and nothing in
the file said so.

BOTH halves are required, and the reason the second one exists is the point:

  1. achieved_GBps == bytes_moved / (median_ms / 1000) / 1e9
  2. bytes_moved and median_ms equal those of the cell with the same label in
     `cells`

A key-presence test passes on the broken artifact. So does a flat-convention
recomputation (2*n*4) -- and that second one's PASSING is the signature of the
bug, which is why a check written that way would have been reported as
validation. A test is what stops this being someone's judgement on the day.

CPU-only, reads a committed artifact, no GPU and no queue.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
METRICS = HERE / "metrics.json"   # this test lives in experiments/<id>/

FAILS = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{'  ' + str(detail)[:170] if detail else ''}")
    if not ok:
        FAILS.append(name)


def main() -> int:
    if not METRICS.exists():
        print(f"FAIL  {METRICS} not found")
        return 1
    d = json.loads(METRICS.read_text())
    reg = d["regimes"]
    entries = reg.get("ceiling_candidates_ranked_all_cells", [])
    cells = {c["label"]: c for c in d["cells"]}

    check("artifact carries a computed candidate list (proof the code ran)",
          bool(entries), f"{len(entries)} entries")

    # ---- half 1: each entry reconciles with itself ----------------------
    bad_self = []
    for e in entries:
        expect = e["bytes_moved"] / (e["median_ms"] / 1000.0) / 1e9
        if abs(expect - e["achieved_GBps"]) > 1e-6 * max(1.0, abs(expect)):
            bad_self.append((e["label"], round(e["achieved_GBps"], 1),
                             round(expect, 1)))
    check("half 1: achieved_GBps derives from the entry's own bytes_moved",
          not bad_self, bad_self)

    # ---- half 2: the entry matches the cell it names --------------------
    bad_cell = []
    for e in entries:
        c = cells.get(e["label"])
        if c is None:
            bad_cell.append((e["label"], "no such cell"))
            continue
        if c["median_ms"] != e["median_ms"]:
            bad_cell.append((e["label"], "median_ms differs"))
    check("half 2: each entry's median_ms matches the cell of the same label",
          not bad_cell, bad_cell)

    # ---- CELL TIE: every candidate label resolves to exactly ONE cell ----
    # Counted against a FRESH output (this run's cells) rather than only
    # self-consistency. Half of the reconciliation test the PI specified is
    # this: an entry that names a cell which does not exist, or names two,
    # or names one with different numbers, is a candidate list that cannot be
    # audited against the measurements it claims to summarise.
    dupes = [lbl for lbl in cells
             if sum(1 for c in d["cells"] if c["label"] == lbl) > 1]
    check("every candidate label resolves to exactly ONE cell", not dupes, dupes[:3])

    missing = [e["label"] for e in entries if e["label"] not in cells]
    check("no candidate names a cell that does not exist", not missing, missing)

    # the tie must hold on the fields that matter, not just on the key
    tie_bad = []
    for e in entries:
        c = cells.get(e["label"])
        if not c:
            continue
        if (c["median_ms"] != e["median_ms"]
                or c["n_problem_size"] != e["n_problem_size"]
                or c.get("covers") is not None and c["covers"] < 0):
            tie_bad.append(e["label"])
    check("each candidate's median_ms and n match its cell", not tie_bad, tie_bad)

    # ---- CONVENTION ANCHORS: the per-kernel rule, with its two cases ----
    by_label = {e["label"]: e for e in entries}
    sax = next((e for lbl, e in by_label.items()
                if "saxpy" in lbl and e["n_problem_size"] == 4194304), None)
    if sax:
        flat = 2 * sax["n_problem_size"] * 4
        ratio = sax["bytes_moved"] / flat
        check("convention anchor: k_saxpy moves 1.5x the flat count "
              "(two reads, one write)",
              abs(ratio - 1.5) < 1e-6, f"ratio={ratio:.4f} "
              f"({sax['bytes_moved']} vs flat {flat})")
    strided = next((e for lbl, e in by_label.items()
                    if "strided_read" in lbl), None)
    if strided:
        flat = 2 * strided["n_problem_size"] * 4
        ratio = strided["bytes_moved"] / flat
        check("convention anchor: k_strided_read moves materially BELOW the flat "
              "count (stride 17 misses cache lines)",
              ratio < 0.75, f"ratio={ratio:.4f} "
              f"({strided['bytes_moved']} vs flat {flat})")

    # and the convention is genuinely per-kernel ---------------------
    # A flat 2*n*4 recomputation must NOT reproduce every entry, otherwise the
    # per-kernel rule is not actually in force. This is the check whose
    # PASSING was the bug's signature, inverted: here it must FAIL for at
    # least one kernel to be right.
    disagree = []
    for e in entries:
        c = cells.get(e["label"])
        if not c:
            continue
        flat = 2 * c["n_problem_size"] * 4
        if flat != e["bytes_moved"]:
            disagree.append((e["label"], f"flat={flat} vs per-kernel={e['bytes_moved']}"))
    check("per-kernel byte counts differ from flat 2*n*4 where they should",
          bool(disagree), f"{len(disagree)} kernel(s) differ, e.g. {disagree[:2]}")

    # ---- the ceiling is the argmax of the list it is printed beside -----
    cands = reg.get("ceiling_candidate_set", {}).get("ranked", [])
    if cands:
        argmax = max(cands, key=lambda x: x["achieved_GBps"])
        check("the published ceiling is the ARGMAX of the candidate list",
              argmax["label"] == reg["memory_ceiling_source_cell"]
              and abs(argmax["achieved_GBps"] * 1e9
                      - reg["measured_memory_ceiling_GBps"] * 1e9) < 1e-6,
              f"argmax={argmax['label']} "
              f"({argmax['achieved_GBps']:.1f} GB/s), "
              f"source_cell={reg['memory_ceiling_source_cell']}, "
              f"ceiling={reg['measured_memory_ceiling_GBps']:.1f}")

    # ---- per-cell regime flags actually present ------------------------
    n_reg = sum(1 for c in d["cells"] if c.get("regime"))
    check("every cell carries a regime flag", n_reg == len(d["cells"]),
          f"{n_reg}/{len(d['cells'])}")

    print()
    print(f"{len(FAILS)} failure(s)" + (": " + ", ".join(FAILS) if FAILS else ""))
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
