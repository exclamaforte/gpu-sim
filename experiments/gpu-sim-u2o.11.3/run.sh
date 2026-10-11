#!/usr/bin/env bash
# gpu-sim-u2o.11.3 -- scale-up verification. CPU-ONLY after the ts measure job.
# Default mode verifies against committed metrics.json and leaves the tree
# untouched; XY8_OUTDIR set to this directory produces it.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
cd "$REPO"
export CUDA_VISIBLE_DEVICES=
OUT="${XY8_OUTDIR:-$(mktemp -d "${XDG_CACHE_HOME:-$HOME/.cache}/u113-XXXXXX")}"
if [ -z "${XY8_OUTDIR:-}" ]; then trap 'rm -rf "$OUT"' EXIT; fi
exec > >(tee "$OUT/output.log") 2>&1

echo "--- 1. submission gate re-runs green on the finished tree ---"
CUDA_VISIBLE_DEVICES= nice -n 10 python3 "$HERE/preflight_static.py" \
    || { echo "FAIL: preflight red"; exit 1; }

echo "--- 2. verify: join, census, noise, contention, split, anchor, item 9 ---"
if [ -n "${XY8_OUTDIR:-}" ]; then
    CUDA_VISIBLE_DEVICES= nice -n 10 python3 "$HERE/verify.py" \
        || { echo "FAIL: verify red"; exit 1; }
else
    CUDA_VISIBLE_DEVICES= nice -n 10 python3 "$HERE/verify.py" \
        --metrics-out "$OUT/metrics.json" \
        || { echo "FAIL: verify red"; exit 1; }
    cmp -s "$HERE/metrics.json" "$OUT/metrics.json" \
        && echo "RECOMPUTE PASS: metrics byte-identical" \
        || { echo "RECOMPUTE FAIL: metrics differ"; exit 1; }
fi

echo "--- 3. frozen references untouched ---"
frozen="experiments/gpu-sim-u2o.7.1/measured.jsonl
experiments/gpu-sim-e5q/measured.jsonl
experiments/gpu-sim-u2o.8.3/features.jsonl"
touched=$(git status --porcelain -- $frozen | head -n 5 || true)
[ -z "$touched" ] || { echo "FAIL: frozen corpus rows touched:"; echo "$touched"; exit 1; }
echo "456 reference rows + 8.3 features: untouched"

echo "--- 4. new tests green; repo-wide nothing new ---"
CUDA_VISIBLE_DEVICES= nice -n 10 python3 tests/test_stage_dim_scalars.py 2>&1 | tail -n 1
CUDA_VISIBLE_DEVICES= nice -n 10 python3 tests/test_regime_census.py 2>&1 | tail -n 1
CUDA_VISIBLE_DEVICES= nice -n 10 python3 tests/test_tree_state_supersession.py 2>&1 | tail -n 3 | head -n 1
CUDA_VISIBLE_DEVICES= nice -n 10 python3 -m pytest tests/ -q -p no:cacheprovider \
    > "$OUT/suite.log" 2>&1 || true
grep -E "^FAILED" "$OUT/suite.log" | sed 's/FAILED //' | sort > "$OUT/failures.txt"
cat "$OUT/failures.txt"
known="tests/test_generated_recipe.py::TestTheRecipeIsGenerated::test_the_generated_rm_lists_every_tracked_path
tests/test_lint_prose_numbers.py::TestFiresOnHistory::test_6_10_seconds_vs_milliseconds
tests/test_lint_prose_numbers.py::TestFiresOnHistory::test_6_11_typed_count
tests/test_lint_prose_numbers.py::TestFiresOnHistory::test_6_6_3_discriminates_ordering
tests/test_lint_prose_numbers.py::TestDoesNotFireOnSupportedClaims::test_6_10_correct_ratio_does_not_fire
tests/test_retro_rows_uncached.py::TestContentCarriesTheClaim::test_the_tree_that_sampled_the_rows_contains_no_cache
tests/test_verify_harness_telemetry.py::TestAgreementIsReal::test_measure_family_gap_is_documented_not_silent"
unknown=$(comm -23 "$OUT/failures.txt" <(echo "$known" | sort))
[ -z "$unknown" ] || { echo "FAIL: new failures:"; echo "$unknown"; exit 1; }
echo "suite: failures subset of known-7; dim-scalars + census + tree_state green"

if [ -n "${XY8_OUTDIR:-}" ]; then echo "PRODUCED in $OUT"; fi
