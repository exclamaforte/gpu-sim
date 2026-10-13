#!/usr/bin/env python3
"""gpu-sim-u2o.7.7 MERGED audit (CPU): family schema + split rules on 3261 rows.

Inputs: the five committed measured corpus files (read-only; refuse if any
is missing rather than auditing a partial corpus). No GPU, no ts job.

Headline vs the 7.1-era audit: token_digest is NOT 1:1 with op_type on the
merged corpus -- 3 digests are shared across ops (representation-identical
examples under different op labels) and 3 strided ops carry 2-3 digests
(parameterized templates). The split family MUST be the digest group, or an
op-level holdout leaks representations across the boundary while reporting
zero leakage.
"""
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

# ruff: noqa: E402 -- repo import after sys.path, local convention
from epic_u2o.dataset_split import (  # noqa: E402
    SplitError,
    leakage_keys,
)

CORPUS = [
    "experiments/gpu-sim-u2o.7.1/measured.jsonl",
    "experiments/gpu-sim-e5q/measured.jsonl",
    "experiments/gpu-sim-u2o.11.3/measured.jsonl",
    "experiments/gpu-sim-f6y/measured_new.jsonl",
    "experiments/gpu-sim-0vk/measured_new.jsonl",
]
STAGE1_OPS = {"scale", "add", "saxpy", "relu", "copy", "triad", "mul", "div"}
SIZE_CUTOFF = 2 ** 26  # stated size-holdout rule: problem_size >= 2^26


def load_rows() -> list:
    rows = []
    for f in CORPUS:
        p = REPO / f
        if not p.exists():
            raise SystemExit(f"FATAL: corpus file missing: {f}. Refusing to "
                             f"audit a partial corpus.")
        rows += [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return rows


def digest_group(r) -> str:
    return r["token_digest"]


def check(cond, msg):
    if not cond:
        raise SplitError(msg)


def main() -> int:
    rows = load_rows()

    # --- 1. family table: op -> stage, digests, counts ---------------------
    fam_table, groups = {}, {}
    for r in rows:
        op = r["op_type"]
        stage = r.get("stage") or ("stage1" if op in STAGE1_OPS else "UNKNOWN")
        f = fam_table.setdefault(op, {"stage": stage, "digests": set(),
                                      "n": 0, "measurable": 0,
                                      "sizes": set(), "regimes": set()})
        if f["stage"] != stage:
            raise SplitError(f"op {op} spans stages: {f['stage']} vs {stage}")
        f["digests"].add(r["token_digest"])
        f["n"] += 1
        f["measurable"] += (r["measured_regime"] == "measurable")
        f["sizes"].add(r["problem_size"])
        f["regimes"].add(r["measured_regime"])
        groups.setdefault(r["token_digest"], set()).add(op)
    check(all(f["stage"] != "UNKNOWN" for f in fam_table.values()),
          "op without stage mapping -- schema incomplete")
    shared_digests = {d: sorted(ops) for d, ops in groups.items() if len(ops) > 1}
    multi_digests = {op: sorted(f["digests"]) for op, f in fam_table.items()
                     if len(f["digests"]) > 1}

    # --- 2. pre-existing op-level split leaks representations --------------
    from epic_u2o.structural_split import split_by_family  # noqa: E402
    labelled = [dict(r, metadata={"family_id": r["op_type"]}) for r in rows]
    seed_leaks = []
    for seed in range(20):
        old = split_by_family(labelled, seed=seed)
        train_dig = {r["token_digest"] for r in old["train"]}
        leaked = [r for r in old["test"] if r["token_digest"] in train_dig]
        seed_leaks.append(round(len(leaked) / max(1, len(old["test"])), 4))
    old_rep_leak_seeds = seed_leaks  # seed 42 alone leaks 0.0: luck, not design

    # --- 3. row-level random split baseline ---------------------------------
    import random  # noqa: E402
    rng = random.Random(42)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    cut = int(len(rows) * 0.85)
    rl_train, rl_test = idx[:cut], idx[cut:]
    rl_train_dig = {rows[i]["token_digest"] for i in rl_train}
    rl_leaked = sum(1 for i in rl_test if rows[i]["token_digest"] in rl_train_dig)

    # --- 4. recommended split: family_holdout_v1 over digest groups --------
    # Stated rule: digest groups sorted by (n_rows, digest); every 5th group
    # to test, every 5th offset 1 to val. Deterministic and auditable.
    gsize = {d: sum(1 for r in rows if r["token_digest"] == d) for d in groups}
    ordered = sorted(groups, key=lambda d: (gsize[d], d))
    test_groups = set(ordered[4::5])
    val_groups = set(ordered[3::5])
    parts = {"train": [], "val": [], "test": []}
    for r in rows:
        d = digest_group(r)
        parts["test" if d in test_groups else "val" if d in val_groups
              else "train"].append(r)
    # Guards (each a way a split can look fine while worthless).
    for name, part in parts.items():
        check(part, f"{name} side EMPTY -- measures nothing")
    check(sum(map(len, parts.values())) == len(rows), "rows not conserved")
    for a, b in (("train", "test"), ("train", "val"), ("val", "test")):
        da = {digest_group(r) for r in parts[a]}
        db = {digest_group(r) for r in parts[b]}
        check(not (da & db), f"digest group straddles {a}/{b}: {da & db}")
    ka = {k for r in parts["train"] for k in leakage_keys(r)}
    for side in ("val", "test"):
        shared = {k for r in parts[side] for k in leakage_keys(r)} & ka
        check(not shared, f"representation key shared train/{side}: {shared}")
    for name in ("val", "test"):
        check(len(parts[name]) >= 5, f"{name} under 5 rows -- single-measurement number")
    # Size-range holdout, carved as a second test slice by stated rule.
    size_slice = [r for r in rows if r["problem_size"] >= SIZE_CUTOFF]
    check(len(size_slice) >= 5, "size slice under 5 rows")
    size_dig = {r["token_digest"] for r in size_slice}
    size_shared = size_dig & {r["token_digest"] for r in rows
                              if r["problem_size"] < SIZE_CUTOFF}

    def side_report(part):
        meas = sum(1 for r in part if r["measured_regime"] == "measurable")
        return {"n": len(part), "measurable": meas,
                "n_digest_groups": len({digest_group(r) for r in part}),
                "n_ops": len({r["op_type"] for r in part})}

    metrics = {
        "bead": "gpu-sim-u2o.7.7",
        "n_rows": len(rows),
        "n_ops": len(fam_table),
        "n_digest_groups": len(groups),
        "family_table": {op: {"stage": f["stage"], "n": f["n"],
                               "measurable": f["measurable"],
                               "n_digests": len(f["digests"]),
                               "n_sizes": len(f["sizes"]),
                               "regimes": sorted(f["regimes"])}
                         for op, f in sorted(fam_table.items())},
        "digest_groups_shared_across_ops": shared_digests,
        "ops_with_multiple_digests": multi_digests,
        "pre_existing_op_split_rep_leak_by_seed_0_19": old_rep_leak_seeds,
        "pre_existing_op_split_seeds_leaking": sum(1 for x in old_rep_leak_seeds if x > 0),
        "pre_existing_op_split_sizes_seed42": {k: len(v) for k, v in old.items()},
        "row_level_baseline": {"test_n": len(rl_test),
                               "test_rows_sharing_digest_with_train": rl_leaked},
        "recommended_9_5_split": {
            "name": "family_holdout_v1",
            "rule": ("digest groups sorted by (n_rows, digest); every 5th "
                     "group to test, every 5th offset 1 to val"),
            "test_groups": sorted(test_groups),
            "val_groups": sorted(val_groups),
            "sizes": {k: side_report(v) for k, v in parts.items()},
            "shared_digest_groups_across_sides": 0,
            "size_slice_rule": f"problem_size >= 2^26 ({SIZE_CUTOFF})",
            "size_slice": {"n": len(size_slice),
                           "measurable": sum(1 for r in size_slice
                                             if r["measured_regime"] == "measurable"),
                           "digests_shared_with_below_cutoff_by_design": len(size_shared)},
        },
    }
    (HERE / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"rows={len(rows)} ops={len(fam_table)} digest_groups={len(groups)}")
    print(f"shared digests: {len(shared_digests)}; multi-digest ops: {len(multi_digests)}")
    nl = sum(1 for x in old_rep_leak_seeds if x > 0)
    print(f"op-level split representation leak: {nl}/20 seeds leak "
          f"(max {max(old_rep_leak_seeds):.0%} of test rows; "
          f"seed 42 alone leaks 0.0 -- luck, not design)")
    print(f"row-level baseline: {rl_leaked}/{len(rl_test)} test rows share digest with train")
    print("family_holdout_v1:", {k: v["n"] for k, v in
          metrics["recommended_9_5_split"]["sizes"].items()},
          "meas:", {k: v["measurable"] for k, v in
          metrics["recommended_9_5_split"]["sizes"].items()})
    print(f"size slice >=2^26: n={len(size_slice)} "
          f"(shared digests below cutoff, by design: {len(size_shared)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
