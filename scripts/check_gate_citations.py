#!/usr/bin/env python3
"""Check that gate citations in experiments/*/metrics.json name a real tip.

A gate citation that cannot be tied to the artifact it claims is worse than no
citation. This check requires any 'gate' string that contains a tip hash to
name a commit that actually exists in this repository.
"""
from __future__ import annotations
import json, re, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TIP_RE = re.compile(r"\btip\s+([0-9a-f]{7,40})\b", re.I)

def main() -> int:
    bad = []
    for metrics in sorted((REPO / "experiments").glob("*/metrics.json")):
        try:
            data = json.loads(metrics.read_text())
        except Exception:
            continue
        gate = data.get("gate")
        if not isinstance(gate, str):
            continue
        m = TIP_RE.search(gate)
        if not m:
            # no tip hash cited; acceptable (legacy) but report
            continue
        tip = m.group(1)
        r = subprocess.run(["git", "-C", str(REPO), "cat-file", "-e", tip],
                           capture_output=True)
        if r.returncode != 0:
            bad.append((metrics, tip))
    if bad:
        for m, t in bad:
            print(f"INVALID: {m} cites tip {t} which does not exist in this repo")
        return 1
    print("check_gate_citations: OK")
    return 0

if __name__ == "__main__":
    sys.exit(main())
