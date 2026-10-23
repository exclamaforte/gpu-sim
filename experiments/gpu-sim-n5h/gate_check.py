"""Default-gate check: with FAKEGPU_NCCL_SIMULATE_ONLY unset, a world-4 DP
sweep must reproduce the COMMITTED 8y7.3 record byte-for-byte on traffic
(coll_msgs=149, coll_MB=86.1 from sweep-zero1.json). step_ms is timing and
may wobble; traffic must not. Reads the sweep table stdout."""
import json
import re
import sys
from pathlib import Path

text = open(sys.argv[1]).read()
rows = [l for l in text.splitlines() if re.match(r"\s*4\s+1\s+1\s+4\s+1\s", l)]
assert len(rows) == 1, f"want exactly one world-4/zero-1 row:\n{text}"
parts = rows[0].split()
coll, mb = int(parts[6]), float(parts[7])
ref = json.load(open(Path(__file__).parent.parent
                      / "gpu-sim-8y7.3" / "sweep-zero1.json"))[0]
print(f"gate: coll={coll} MB={mb} (record {ref['coll_msgs']} / {ref['coll_MB']})")
assert coll == ref["coll_msgs"], f"collective count drifted: {coll}"
assert abs(mb - ref["coll_MB"]) < 0.05, f"traffic drifted: {mb}"
print("GATE PASS: default behaviour reproduces the committed record")
