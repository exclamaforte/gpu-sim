#!/usr/bin/env python3
"""Lint: a radius/flip claim must not be rendered as a two-sided bound unless
both sides were actually measured (gpu-sim-b5f2)."""
from __future__ import annotations
import re, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOC = REPO / "docs" / "training-parallelism-sim.md"

TWO_SIDED = re.compile(r"unchanged for .* in \[")
bad = []
text = DOC.read_text()
for line in text.splitlines():
    if TWO_SIDED.search(line):
        if "measured" not in line.lower():
            bad.append(line.strip())
if bad:
    for b in bad:
        print(f"INVALID (no measured_sides): {b}")
    sys.exit(1)
print("check_radius_directions: OK")
sys.exit(0)
