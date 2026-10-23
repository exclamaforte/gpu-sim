"""Histogram a GPUSIM trace by (rank, op, algorithm, regime)."""
import json
import sys
from collections import Counter

c: Counter = Counter()
n = 0
for line in open(sys.argv[1]):
    line = line.strip()
    if not line:
        continue
    e = json.loads(line)
    c[(e.get("rank"), e.get("op"), e.get("algorithm"), e.get("regime"))] += 1
    n += 1
print("lines:", n)
for k, v in sorted(c.items()):
    print(v, k)
