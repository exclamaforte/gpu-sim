#!/usr/bin/env python3
"""Run the split design against the REAL 7.1 measured dataset. 7.7.

No GPU and no ts job: the input is 7.1's committed measurement artifact
(ts 20260928t074004-b429f469), consumed read-only. Re-measuring would not
change anything about how a split must be built.
"""
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from epic_u2o.dataset_split import (  # noqa: E402
    SplitError, family_disjoint_split, leakage_audit, structural_family,
    validate_split,
)

DATASET_REF = "lab/gpu-sim-u2o.7.1"
DATASET = "experiments/gpu-sim-u2o.7.1/measured.jsonl"


def load_rows() -> list:
    """Read 7.1's committed rows. Refuse rather than substitute a local copy."""
    r = subprocess.run(["git", "show", f"{DATASET_REF}:{DATASET}"],
                       cwd=REPO, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(
            f"FATAL: cannot read {DATASET_REF}:{DATASET}\n{r.stderr}\n"
            f"This experiment reports on the REAL measured dataset. A synthetic "
            f"stand-in would make every number below a fiction.")
    return [json.loads(l) for l in r.stdout.splitlines() if l.strip()]


def main() -> int:
    rows = load_rows()
    fams = sorted({structural_family(r) for r in rows})
    sizes = sorted({r["problem_size"] for r in rows})
    regimes = Counter(r["measured_regime"] for r in rows)

    out: dict = {
        "bead": "gpu-sim-u2o.7.7",
        "dataset": f"{DATASET_REF}:{DATASET}",
        "dataset_ts_job": "20260928t074004-b429f469",
        "n_rows": len(rows),
        "families": fams,
        "n_families": len(fams),
        "problem_sizes": sizes,
        "measured_regimes": dict(regimes),
    }

    # --- what the pre-existing split logic does to this data ---------------
    from epic_u2o.structural_split import (
        analyze_split_quality, assign_family, split_by_family)
    try:
        fam_none = assign_family(rows[0], rows[0]["op_type"])
    except Exception as exc:
        fam_none = f"RAISED {type(exc).__name__}: {exc}"
    old = split_by_family(rows, seed=42)
    old_q = analyze_split_quality(old)
    out["pre_existing_structural_split"] = {
        "assign_family_returns": fam_none,
        "split_sizes": {k: len(v) for k, v in old.items()},
        "quality_reported_total_families": old_q.get("total_families"),
        "quality_reported_leakage_train_test": old_q.get("leakage_train_test"),
        "verdict": ("FAIL-OPEN: every row went to train, val and test are empty, "
                    "and the leakage analyzer reported 0 families and 0 leakage. "
                    "A check that reports 'no leakage' for a split with no test "
                    "set is not a weak check, it is a green result produced by "
                    "nothing having been tested."),
    }

    # --- the representation finding ----------------------------------------
    digests = {r["token_digest"] for r in rows}
    per_digest = Counter(r["token_digest"] for r in rows)
    digest_to_fams = {}
    for r in rows:
        digest_to_fams.setdefault(r["token_digest"], set()).add(r["op_type"])
    digest_invariant_to_size = all(
        len({r["problem_size"] for r in rows if r["token_digest"] == d}) == 1
        for d in digests)
    digest_invariant_to_launch = all(
        len({json.dumps(r["launch_config"], sort_keys=True)
             for r in rows if r["token_digest"] == d}) == 1
        for d in digests)
    out["representation_finding"] = {
        "n_distinct_token_digests": len(digests),
        "rows_per_digest": sorted(set(per_digest.values())),
        "digests_mapping_to_more_than_one_family": sum(
            1 for v in digest_to_fams.values() if len(v) > 1),
        "digest_invariant_to_problem_size": digest_invariant_to_size,
        "digest_invariant_to_launch_config": digest_invariant_to_launch,
        "what_it_means": (
            "The tokenizer emits ONE distinct sequence per family and it does not "
            "vary with problem_size or launch_config. So 264 measured rows "
            "collapse to 8 distinct model inputs. A row-level split therefore "
            "puts token-identical examples on both sides, and a model asked to "
            "predict runtime from this representation cannot see the two things "
            "that determine runtime. This is a gpu-sim-u2o.8 finding that only "
            "became visible once the split was run on real data."),
    }

    # --- what a row-level split would have claimed --------------------------
    out["leakage_audit"] = leakage_audit(rows, seed=42)

    # --- the split this bead recommends ------------------------------------
    # 8 families -> hold out 2 for test and 1 for val, deterministically and
    # by a stated rule, so the choice is auditable rather than lucky.
    ordered = sorted(fams, key=lambda f: (f,))
    test_fams, val_fams = ordered[:2], ordered[2:3]
    split = family_disjoint_split(rows, test_families=test_fams,
                                  val_families=val_fams)
    out["recommended_split"] = {
        "rule": "families sorted alphabetically; first 2 to test, next 1 to val. "
                "A stated rule rather than a random draw, so the split is "
                "reproducible and auditable instead of lucky.",
        "test_families": test_fams,
        "val_families": val_fams,
    }
    try:
        out["recommended_split"]["validation"] = validate_split(split, rows)
    except SplitError as exc:
        out["recommended_split"]["validation"] = {"FAILED": str(exc)}
        out["recommended_split"]["passed"] = False
    else:
        out["recommended_split"]["passed"] = True

    # A size-range holdout is impossible here and saying so is the finding.
    out["size_range_holdout"] = {
        "possible": False,
        "why": (f"the dataset has {len(sizes)} problem sizes ({sizes}) and every "
                f"one of the {len(rows)} rows is measured_regime="
                f"{list(regimes)[0]!r}. A size-range holdout needs sizes spanning "
                f"more than one regime; here every size is launch-bound (7.1's "
                f"finding), so holding out a size range would hold out a regime "
                f"change that does not exist in the data. Recorded as impossible "
                f"rather than implemented and reported as if it worked."),
    }

    (HERE / "metrics.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in
                      ("n_rows", "n_families", "problem_sizes")}, indent=2))
    print("\npre-existing split:",
          json.dumps(out["pre_existing_structural_split"]["split_sizes"]))
    print("  it reported leakage_train_test =",
          out["pre_existing_structural_split"]["quality_reported_leakage_train_test"])
    print("\nrepresentation:",
          out["representation_finding"]["n_distinct_token_digests"],
          "distinct token digests for", out["n_rows"], "rows")
    a = out["leakage_audit"]["row_level_split"]
    print(f"row-level split: {a['test_rows_leaked_fraction']:.1%} of test rows "
          f"share a representation key with train; "
          f"{a['n_families_shared']}/{a['n_families_total']} families shared")
    print("recommended split passes validation:",
          out["recommended_split"]["passed"])
    return 0 if out["recommended_split"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
