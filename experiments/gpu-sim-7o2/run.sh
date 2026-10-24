#!/usr/bin/env bash
# gpu-sim-7o2: recompute the timing summary from the committed reps and diff
# against the committed summary.json. Recomputation goes to a scratch path;
# running this leaves `git status` clean.
set -u
cd /home/gabe/Documents/gpu-sim/.lab/wt/gpu-sim-7o2
D=experiments/gpu-sim-7o2
CUDA_VISIBLE_DEVICES= python3 - "$D/reps_probe.jsonl" "$D/reps_main.jsonl" > /tmp/7o2-recompute.txt <<'EOF'
import json, statistics, sys
cells = {}
for p in sys.argv[1:]:
    for line in open(p):
        if line.strip():
            r = json.loads(line)
            assert r.get("rc") == 0 and r.get("step_ms") is not None, r
            cells.setdefault(r["cell"], []).append(r["step_ms"])
out = {}
for name in sorted(cells):
    xs = sorted(cells[name])
    assert len(xs) >= 7, (name, len(xs))
    q1, med, q3 = statistics.quantiles(xs, n=4, method="exclusive")[0], statistics.median(xs), statistics.quantiles(xs, n=4, method="exclusive")[2]
    out[name] = {"n": len(xs), "median_ms": round(med, 1),
                 "q1_ms": round(q1, 1), "q3_ms": round(q3, 1),
                 "min_ms": round(xs[0], 1), "max_ms": round(xs[-1], 1)}
print(json.dumps(out, indent=2, sort_keys=True))
EOF
diff /tmp/7o2-recompute.txt <(python3 -c "import json; print(json.dumps(json.load(open('$D/summary.json')), indent=2, sort_keys=True))") \
    && echo "7o2 REPRODUCIBLE: summary.json matches recomputation from committed reps"
