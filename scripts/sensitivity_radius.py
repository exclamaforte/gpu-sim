#!/usr/bin/env python3
"""How wrong can the least-validated terms be before the recommendation changes? (gpu-sim-radius)

The document makes decisive claims at N=8 and N=16, and in each case the decisive term is the one the
harness cannot check:

  N=8  (tp=4 pp=1 cp=2 dp=1)  -> CP. The CP axis is un-reconcilable BY CONSTRUCTION (the model prices a
                                 ring p2p exchange, the harness runs differentiable all_gather+RS), and a
                                 4x error was already found in it once (gpu-sim-ncb6/kvcp).
  N=16 (tp=1 pp=16 cp=1 dp=1) -> the PP BUBBLE. No measurement covers it at all; the model assumes
                                 micro_batches = pp because nothing tells it otherwise.

A margin of 6.45% is only meaningful next to "unchanged for term error in [a, b]". This script computes
that interval by bisection with the PRODUCTION model (hw["term_scale"], default 1.0 no-op), so it cannot
drift from the thing it describes.

Usage: python3 scripts/sensitivity_radius.py [--json]
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from epic_cluster import parallelism_search as ps  # noqa: E402

PEAK = 989e12
SPECS = {8: "specs/h100_rack.json", 16: "specs/h100_rack.json", 32: "specs/h100_rack_4node.json"}
GRID = [(8, 8192), (16, 8192), (32, 8192), (8, 32768)]


def hw(n: int, term: str | None = None, k: float = 1.0) -> dict:
    h = {"peak_flops": PEAK, "mem_gb": 80.0, "spec_path": SPECS[n]}
    if term is not None:
        h["term_scale"] = {term: k}
    return h


def winner(n: int, seq: int, term: str | None = None, k: float = 1.0):
    """Winner under a term error k, at the row's own sequence length."""
    h = hw(n, term, k)
    h["seq"] = seq
    _, feas = ps.search(n, h, dict(ps.MODEL, seq=seq))
    return feas[0]


def shape(c) -> tuple:
    return (c.tp, c.pp, c.cp, c.dp, c.zero)


def flip_points(n: int, seq: int, term: str, lo: float = 0.05, hi: float = 8.0):
    """Return (flip_up, flip_down) for a term.

    flip_up: smallest k > 1 at which the winner changes, or None.
    flip_down: largest k < 1 at which the winner changes, or None.
    Bisection in log space.
    """
    import math
    base = shape(winner(n, seq))
    flip_up = None
    flip_down = None
    # upward
    if shape(winner(n, seq, term, hi)) != base:
        a, b = 1.0, hi
        for _ in range(60):
            mid = (a + b) / 2
            if shape(winner(n, seq, term, mid)) == base:
                a = mid
            else:
                b = mid
        flip_up = round(b, 4)
    # downward
    if shape(winner(n, seq, term, lo)) != base:
        a, b = lo, 1.0
        for _ in range(60):
            mid = (a + b) / 2
            if shape(winner(n, seq, term, mid)) == base:
                b = mid
            else:
                a = mid
        flip_down = round(a, 4)
    return flip_up, flip_down


def main() -> int:
    out = {}
    for n, seq in GRID:
        base = winner(n, seq)
        row = {"winner": list(shape(base)), "step_ms": round(base.step_sum * 1e3, 2)}
        for term in ("cp", "pp"):
            up, down = flip_points(n, seq, term)
            row[f"flip_{term}_up"] = up
            row[f"flip_{term}_down"] = down
            sides = []
            if up is not None:
                sides.append("up")
            if down is not None:
                sides.append("down")
            row[f"measured_sides_{term}"] = sides
            if term == "cp":
                row["winner_at_high_cp"] = list(shape(winner(n, seq, term, 8.0)))
        out[f"N{n}_seq{seq}"] = row
    if "--json" in sys.argv:
        print(json.dumps(out, indent=2, sort_keys=True))
        return 0
    print("Robustness radius: how large a multiplicative error in one term before the winner flips")
    print(f"{'cell':16s} {'winner':22s} {'cp_up':>9s} {'cp_down':>9s} {'pp_up':>9s} {'pp_down':>9s}")
    for key, row in out.items():
        w = "".join(str(x) for x in row["winner"])
        cpu = row["flip_cp_up"] if row["flip_cp_up"] is not None else ">8"
        cpd = row["flip_cp_down"] if row["flip_cp_down"] is not None else "<.05"
        ppu = row["flip_pp_up"] if row["flip_pp_up"] is not None else ">8"
        ppd = row["flip_pp_down"] if row["flip_pp_down"] is not None else "<.05"
        print(f"{key:16s} {w:22s} {str(cpu):>9s} {str(cpd):>9s} {str(ppu):>9s} {str(ppd):>9s}")
    print()
    print("Reading it: the N=8 verdict survives a CP error up to the listed factor and no further;")
    print("k > 8 means the term is not decisive at that cell at all.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
