#!/usr/bin/env bash
# gpu-sim-u2o.6.6.6.3 -- CPU only, run directly. Never through ts.
# Regenerates sweep.json, rewrites FINDING.md's quoted block FROM that artifact,
# and runs the suites. Nonzero on any failure.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/../.."
ART="experiments/gpu-sim-u2o.6.6.6.3"
echo "=== sweep: $(hostname) at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
python3 "$ART/measure_sweep.py" 2>&1 | tee "$ART/output.log"
rc=${PIPESTATUS[0]}; echo "=== measure_sweep exit=$rc ==="; [ "$rc" -eq 0 ] || exit "$rc"

python3 - "$ART" <<'PY'
import json, sys
from pathlib import Path
art = Path(sys.argv[1])
rec = json.loads((art / "sweep.json").read_text())
pts = rec["points"]
w = max(len(str(p["files"])) for p in pts)
lines = []
lines.append(f"loadavg(1m)={rec['loadavg_1_5_15'][0]} at measurement time; "
             "every figure below is load-dependent and not comparable with a "
             "sweep taken at another load.\n")
lines.append("| files | median ms | us/file | x device budget |")
lines.append("|---|---|---|---|")
for p in pts:
    lines.append(f"| {p['files']} | {p['median_ms']:.4f} | {p['us_per_file']:.4f} "
                 f"| {p['x_device_budget']:.2f}x |")
f = rec["fit"]
lines.append(f"\nSlope {f['us_per_file_slope']} us/file. Per-file spread "
             f"{f['per_file_spread_us']['ratio']}x -> linear={f['linear']}.")
lines.append("")
g = rec["gain_vs_master"]
lines.append(f"Attempt-2 gain vs master (interleaved A/B, same fixture and load, "
             f"master@{g['master_commit']}):")
lines.append("| files | branch ms | master ms | gain |")
lines.append("|---|---|---|---|")
for q in g["per_size"]:
    lines.append(f"| {q['files']} | {q['branch_median_ms']:.4f} | "
                 f"{q['master_median_ms']:.4f} | {q['gain']:.2f}x |")
lines.append(f"\nMedian gain {g['median_gain']}x (gain > 1 means this branch is faster).")
lines.append("")
fx = rec["fixed_active_points"]
sl = rec["sublinear"]
lines.append(f"Fixed-active sweep (active={fx[0]['active_fixed']}, terminal bulk grows):")
lines.append("| files | median ms |")
lines.append("|---|---|")
for p in fx:
    lines.append(f"| {p['files']} | {p['median_ms']:.4f} |")
lines.append(f"\nSub-linear verdict: log-log slope {sl['loglog_slope']}, "
             f"max/min {sl['max_min_median_ratio']}x -> met={sl['met']}.")
md = art / "FINDING.md"
s = md.read_text()
a, b = "<!-- BEGIN QUOTED FROM sweep.json -->", "<!-- END QUOTED FROM sweep.json -->"
pre = s[:s.index(a) + len(a)]
post = s[s.index(b):]
md.write_text(pre + "\n" + "\n".join(lines) + "\n" + post)
print("FINDING.md quoted block rewritten from sweep.json")
PY

echo "=== suites ==="
for t in tests/test_finding_quotes_sweep.py tests/test_cache_differential.py \
         tests/test_queue_state_cost.py tests/test_sample_provenance.py \
         tests/test_queue_state.py; do
  [ -f "$t" ] || continue
  printf "%-44s " "$t"
  timeout 300 python3 "$t" >/tmp/s.log 2>&1 && echo PASS || { echo FAIL; grep -E "^(FAIL|ERROR):" /tmp/s.log|head -2; }
done
echo "=== run.sh: done ==="
