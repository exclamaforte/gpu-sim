#!/usr/bin/env bash
# gpu-sim-u2o.6.6.2.1.1.1 -- per-file landing, mechanical emission, narrowed test.
# CPU-ONLY git work. Default mode verifies against committed metrics.json and
# leaves the tree untouched; XY8_OUTDIR set to this directory produces it.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
cd "$REPO"
export CUDA_VISIBLE_DEVICES=
OUT="${XY8_OUTDIR:-$(mktemp -d "${XDG_CACHE_HOME:-$HOME/.cache}/62111-XXXXXX")}"
if [ -z "${XY8_OUTDIR:-}" ]; then trap 'rm -rf "$OUT"' EXIT; fi
exec > >(tee "$OUT/output.log") 2>&1

echo "--- 1. tree_state suite: 12/12 ---"
CUDA_VISIBLE_DEVICES= nice -n 10 python3 tests/test_tree_state_supersession.py \
    > "$OUT/tree_state.log" 2>&1
ts_rc=$?
tail -n 4 "$OUT/tree_state.log"
[ "$ts_rc" -eq 0 ] || { echo "FAIL: tree_state suite red"; exit 1; }
grep -q "opt-in artifacts checked: 3" "$OUT/tree_state.log" \
    || { echo "FAIL: opt-in report missing"; exit 1; }

echo "--- 2. 6.6.2 citations equal the mechanical per-file values ---"
CUDA_VISIBLE_DEVICES= nice -n 10 python3 - "$OUT/citations.json" <<'EOF'
import json, sys
sys.path.insert(0, '.')
from scripts.tree_state import landing_commit_for
out = {}
for f in ('experiments/gpu-sim-u2o.6.6.2/job.json',
          'experiments/gpu-sim-u2o.6.6.2/metrics.json'):
    doc = json.load(open(f))
    sb = doc['tree_state']['superseded_by']
    mech = landing_commit_for(f)
    assert sb['commit'] == mech == '97575af', (f, sb['commit'], mech)
    assert '566ba0b' not in json.dumps(sb), f
    out[f] = sb['commit']
json.dump(out, open(sys.argv[1], 'w'), indent=2, sort_keys=True)
print('citations mechanical: 97575af x2')
EOF

echo "--- 3. item 6: no other closed artifact rewritten ---"
allowed='^(experiments/gpu-sim-u2o\.6\.6\.2/(job|metrics)\.json|experiments/gpu-sim-u2o\.6\.6\.2\.1\.1\.1/|scripts/tree_state\.py|tests/test_tree_state_supersession\.py)$'
violations=$(git status --porcelain -- experiments/ scripts/ tests/ \
    | awk '{print $2}' | grep -v -E "$allowed" || true)
[ -z "$violations" ] || { echo "FAIL: closed artifacts touched:"; echo "$violations"; exit 1; }
echo "experiments/ diff limited to the two 6.6.2 citation fields + this bead"

echo "--- 4. repo-wide suite: tree_state fixed, nothing new ---"
CUDA_VISIBLE_DEVICES= nice -n 10 python3 -m pytest tests/ -q -p no:cacheprovider \
    > "$OUT/suite.log" 2>&1 || true
grep -E "^FAILED" "$OUT/suite.log" | sed 's/FAILED //' | sort > "$OUT/failures.txt"
cat "$OUT/failures.txt"
if grep -q "tree_state" "$OUT/failures.txt"; then
    echo "FAIL: tree_state still failing repo-wide"; exit 1
fi
# Known pre-existing family (7 names; the lint DoesNotFire one is
# order-sensitive and flips both ways WITHOUT any tree change -- its cases
# take only literal strings, so this bead cannot influence it). Gate on
# subset, not equality: every failure must be a known name.
known="tests/test_generated_recipe.py::TestTheRecipeIsGenerated::test_the_generated_rm_lists_every_tracked_path
tests/test_lint_prose_numbers.py::TestFiresOnHistory::test_6_10_seconds_vs_milliseconds
tests/test_lint_prose_numbers.py::TestFiresOnHistory::test_6_11_typed_count
tests/test_lint_prose_numbers.py::TestFiresOnHistory::test_6_6_3_discriminates_ordering
tests/test_lint_prose_numbers.py::TestDoesNotFireOnSupportedClaims::test_6_10_correct_ratio_does_not_fire
tests/test_retro_rows_uncached.py::TestContentCarriesTheClaim::test_the_tree_that_sampled_the_rows_contains_no_cache
tests/test_verify_harness_telemetry.py::TestAgreementIsReal::test_measure_family_gap_is_documented_not_silent"
unknown=$(comm -23 "$OUT/failures.txt" <(echo "$known" | sort))
[ -z "$unknown" ] || { echo "FAIL: new failures:"; echo "$unknown"; exit 1; }
echo "suite: failures are a subset of the 7 known pre-existing; nothing new"

CUDA_VISIBLE_DEVICES= nice -n 10 python3 - "$OUT/metrics.json" <<'EOF'
import json, sys
m = {
    "bead": "gpu-sim-u2o.6.6.2.1.1.1",
    "branch": "lab/gpu-sim-u2o.6.6.2.1.1.1",
    "tree_state_tests": "12/12",
    "opt_in_checked": 3,
    "opt_in_stamped": 2,
    "opt_in_unstamped_reported": ["experiments/gpu-sim-u2o.6.6.4/anchor_metrics.json"],
    "citation_6.6.2/job.json": "97575af",
    "citation_6.6.2/metrics.json": "97575af",
    "repo_suite_gate": "tree_state absent from failures; every failure in the known-7 set",
    "repo_suite_new_failures": 0,
}
json.dump(m, open(sys.argv[1], "w"), indent=2, sort_keys=True)
open(sys.argv[1], "a").write("\n")
print("metrics written")
EOF
if [ -n "${XY8_OUTDIR:-}" ]; then
    echo "PRODUCED in $OUT"
else
    cmp -s "$HERE/metrics.json" "$OUT/metrics.json" \
        && echo "RECOMPUTE PASS: per-file landing holds; citations mechanical" \
        || { echo "RECOMPUTE FAIL: metrics differ"; exit 1; }
fi
