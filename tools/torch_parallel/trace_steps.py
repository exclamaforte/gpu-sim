"""Bucket a GPUSIM trace into steps by time gaps, count ops per bucket."""
import json
import sys

evs = []
for line in open(sys.argv[1]):
    line = line.strip()
    if line:
        evs.append(json.loads(line))
evs.sort(key=lambda e: e["ts_ns"])
# split on gaps > 20% of total span (steps are ms apart, intra-step ops us apart)
span = evs[-1]["ts_ns"] - evs[0]["ts_ns"]
buckets, cur, prev = [], [], None
for e in evs:
    if prev is not None and e["ts_ns"] - prev > span * 0.05:
        buckets.append(cur)
        cur = []
    cur.append(e)
    prev = e["ts_ns"]
buckets.append(cur)
print(f"lines={len(evs)} span_ms={span/1e6:.2f} buckets={len(buckets)}")
for i, b in enumerate(buckets):
    from collections import Counter
    c = Counter((e["rank"], e["op"]) for e in b)
    b0, b1 = b[0]["ts_ns"], b[-1]["ts_ns"]
    print(f"bucket{i}: n={len(b)} width_us={(b1-b0)/1e3:.1f} {dict(c)}")
