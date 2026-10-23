"""tbp gate: (1) world-2 --dp-mean loss CURVE matches world-1 to fp tolerance
at every step; (2) the 50-step world-2 curve actually falls. Compares summary
JSONs (which now carry the full per-step `losses` series), never trace vs
trace. NOTE on (1): per-rank losses are LOCAL-shard means, so ranks legitimately
differ; the comparable quantity is the MEAN across ranks (equal shards ->
exact full-batch mean in exact arithmetic), checked against world-1.
Usage: tbp_check.py <world1.json> <world2.json> [--long <world2-50.json>]"""
import json
import sys


def load(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    assert rows, f"no rows in {path}"
    return rows


def mean_curve(rows, tag):
    s = [r["losses"] for r in rows]
    n = len(s[0])
    assert all(len(x) == n for x in s), f"{tag}: ragged rank series"
    return [sum(vals) / len(vals) for vals in zip(*s)]


w1 = mean_curve(load(sys.argv[1]), "world1")
w2 = mean_curve(load(sys.argv[2]), "world2")
assert len(w1) == len(w2), f"length {len(w1)} vs {len(w2)}"
worst = 0.0
for i, (a, b) in enumerate(zip(w1, w2)):
    tol = 1e-5 * max(1.0, abs(a))
    d = abs(a - b)
    worst = max(worst, d / max(1e-12, abs(a)))
    assert d <= tol, f"step {i}: world1={a:.6f} world2mean={b:.6f} (tol {tol:.2e})"
print(f"EQUIV PASS: {len(w1)} steps match, worst rel err {worst:.2e}")
print(f"  world1 {w1[0]:.4f} -> {w1[-1]:.4f} | world2 {w2[0]:.4f} -> {w2[-1]:.4f}")

if len(sys.argv) > 4 and sys.argv[3] == "--long":
    long = mean_curve(load(sys.argv[4]), "world2-long")
    assert long[-1] < long[0], f"no fall: {long[0]:.4f} -> {long[-1]:.4f}"
    ups = sum(1 for a, b in zip(long, long[1:]) if b > a)
    assert ups <= len(long) // 5, f"{ups} up-steps of {len(long)}"
    print(f"LONG PASS: {long[0]:.4f} -> {long[-1]:.4f} over {len(long)} steps, "
          f"{ups} up-steps")
