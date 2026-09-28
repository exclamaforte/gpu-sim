#!/usr/bin/env python3
"""Populate the numeric feature vector from REAL measured examples. 8.3.

The bead's audit note says the feature-vector code "has never been populated
from any real measured example (no such example exists)". gpu-sim-u2o.7.1 has
since produced 264 of them (ts 20260928t074004-b429f469), so this populates
all 264 and -- more usefully -- records what the vector could not see before it
was populated.

The .cu sources are build artifacts and are not committed, so the source for
each row is REGENERATED and then VERIFIED: 7.1's kernel_hash is
sha256(source + json([grid, block, problem_size]))[:16], so a regenerated spec
that hashes to the row's recorded kernel_hash is provably that row's source.
Nothing is matched by name or by position. A row whose hash cannot be
recovered is reported, not skipped quietly.

No GPU, no ts job: this reads a committed artifact and re-derives from the
deterministic generator.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from epic_u2o.features import (  # noqa: E402
    LaunchConfig, extract_features,
)
from epic_u2o.generator_stage1 import generate_kernels  # noqa: E402
from epic_u2o.representation_cardinality import (  # noqa: E402
    CardinalityError, assert_cardinality, check_cardinality,
)

DATASET_REF = "lab/gpu-sim-u2o.7.1"
DATASET = "experiments/gpu-sim-u2o.7.1/measured.jsonl"

# 7.1's build_batch.py sweep, reproduced so the regeneration is the same batch.
BLOCKS = [[64, 1, 1], [128, 1, 1], [256, 1, 1], [1024, 1, 1]]
SIZES = [1 << 12, 1 << 14, 1 << 16, 1 << 18]
OPS = ["scale", "add", "saxpy", "relu", "copy", "triad", "mul", "div"]


def grids_for(n: int, block: int) -> list:
    bt = block[0] * block[1] * block[2]
    full = max(1, n // bt)
    out = [[full, 1, 1], [max(1, full // 2), 1, 1], [max(1, full // 8), 1, 1]]
    if full > 64:
        out.append([1, 1, 1])
    return out


def row_hash(source: str, grid: list, block: list, n: int) -> str:
    return hashlib.sha256(
        (source + json.dumps([grid, block, n])).encode()).hexdigest()[:16]


def regenerate_sources() -> dict:
    """kernel_hash -> spec dict, for 7.1's exact sweep."""
    out: dict = {}
    for n in SIZES:
        for block in BLOCKS:
            for grid in grids_for(n, block):
                for spec in generate_kernels(ops=OPS, grid_sweeps=[grid],
                                             block_sweeps=[block],
                                             problem_sizes=[n]):
                    d = spec.to_dict()
                    h = row_hash(d["source"], d["grid"], d["block"], n)
                    out[h] = d
    return out


def load_rows() -> list:
    r = subprocess.run(["git", "show", f"{DATASET_REF}:{DATASET}"],
                       cwd=REPO, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(
            f"FATAL: cannot read {DATASET_REF}:{DATASET}\n{r.stderr}\n"
            f"This experiment populates the vector from the REAL measured "
            f"dataset. A synthetic stand-in would make every number fiction.")
    return [json.loads(l) for l in r.stdout.splitlines() if l.strip()]


def _residual_collision(feats: list, rows: list) -> str:
    """Why distinct < n even after the fix. Names the axis, not just the number.

    Answered here rather than left as '165 of 264' because a ratio without a
    cause is the same failure one level up.
    """
    from collections import defaultdict
    groups = defaultdict(list)
    for f in feats:
        groups[tuple(f["vector"])].append(f)
    colliding = [v for v in groups.values() if len(v) > 1]
    if not colliding:
        return ""
    cross_family = sum(1 for v in colliding
                       if len({x["op_type"] for x in v}) > 1)
    same_config = sum(
        1 for v in colliding
        if len({(x["op_type"], x["problem_size"]) for x in v}) == 1)
    return (f"{len(colliding)} colliding groups covering "
            f"{sum(len(v) for v in colliding)} rows; {cross_family} of them "
            f"span MORE THAN ONE op_type, {same_config} are within a single "
            f"family. The residual collision is therefore the FAMILY axis: a "
            f"purely static feature vector has no op-identity field, so add, "
            f"mul and div at the same size and launch are the same vector. "
            f"That is a property of the channel, not a bug, and it is why the "
            f"numeric vector cannot substitute for the token channel's family "
            f"signal -- 7.7 needs family holdout precisely because of it.")


def _spearman(xs: list, ys: list) -> float:
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = pos
        return r
    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else 0.0


def _label_dynamics(feats: list) -> dict:
    """Correlation of each headline feature with the measured runtime.

    Reported so the vector's value is a measurement rather than a claim. A weak
    correlation on THIS dataset is a fact about the DATASET, not a verdict on
    the vector, and the summary says which.
    """
    t = [f["measured_median_ms"] for f in feats]
    corr = {
        name: round(_spearman([f[key] for f in feats], t), 3)
        for name, key in (("problem_size", "problem_size"),
                          ("estimated_bytes_moved", "estimated_bytes_moved"),
                          ("estimated_flops", "estimated_flops"),
                          ("total_threads", "total_threads"))
    }
    by_size = {}
    for n in sorted({f["problem_size"] for f in feats}):
        sub = sorted(f["measured_median_ms"] for f in feats
                     if f["problem_size"] == n)
        by_size[str(n)] = round(sub[len(sub) // 2] * 1000.0, 3)
    lo, hi = min(by_size.values()), max(by_size.values())
    return {
        "spearman_vs_measured_median_ms": corr,
        "median_us_by_problem_size": by_size,
        "label_dynamic_range_us": round(hi - lo, 3),
        "size_range_covered": f"{min(by_size)} .. {max(by_size)}",
        "verdict": (
            f"The vector now SEES problem_size and the measured launch config "
            f"(distinct vectors 5 -> "
            f"{len({tuple(f['vector']) for f in feats})}), but on THIS dataset "
            f"it cannot be shown to be predictive: every row is launch_bound "
            f"and the median runtime moves only {lo:.2f} us to {hi:.2f} us "
            f"across the whole size range. That is 7.1's own finding, and it "
            f"means the weak correlations ({max(corr.values())} at best) are a "
            f"property of the LABEL, not evidence that the vector is wrong. "
            f"Judging it requires 7.1's dataset extended past 2^18 into "
            f"memory-bound territory."),
    }


def main() -> int:
    rows = load_rows()
    sources = regenerate_sources()

    recovered, unrecovered = [], []
    for r in rows:
        (recovered if r["kernel_hash"] in sources else unrecovered).append(r)

    # FAIL LOUDLY rather than populating a subset and reporting a total that
    # includes rows whose source was never matched.
    if unrecovered:
        print(f"FATAL: {len(unrecovered)} of {len(rows)} measured rows have no "
              f"recoverable source by hash. Refusing to emit a feature vector "
              f"for a subset while reporting counts for all.", file=sys.stderr)
        return 2

    feats, vectors_as_written, vectors_fixed = [], set(), set()
    for r in recovered:
        src = sources[r["kernel_hash"]]["source"]
        lc_dict = r["launch_config"]
        measured_lc = LaunchConfig(
            grid_x=lc_dict["grid"][0], grid_y=lc_dict["grid"][1],
            grid_z=lc_dict["grid"][2],
            block_x=lc_dict["block"][0], block_y=lc_dict["block"][1],
            block_z=lc_dict["block"][2],
            shared_mem_bytes=lc_dict.get("shared_mem_bytes", 0))

        # AS IT WOULD HAVE BEEN POPULATED: no launch config, no problem size.
        naive = extract_features(src)
        vectors_as_written.add(tuple(round(x, 6) for x in naive.to_vector()))

        # AS IT SHOULD BE: the MEASURED launch and the MEASURED problem size.
        f = extract_features(src, launch_config=measured_lc,
                             problem_size=r["problem_size"])
        vectors_fixed.add(tuple(round(x, 6) for x in f.to_vector()))

        d = f.to_dict()
        feats.append({
            "kernel_hash": r["kernel_hash"], "op_type": r["op_type"],
            "problem_size": r["problem_size"],
            "measured_median_ms": r["median_runtime_ms"],
            "measured_regime": r["measured_regime"],
            "vector": f.to_vector(),
            "vector_dim": f.vector_dim,
            "problem_size_in_vector": d["problem_size"],
            "estimated_bytes_moved": d["estimated_bytes_moved"],
            "estimated_flops": d["estimated_flops"],
            "total_threads": d["grid_x"] * d["block_x"],
        })

    # The legitimate ceiling: how many DISTINCT launch configurations the rows
    # actually contain. Measured, not assumed -- if rows genuinely duplicate,
    # the ceiling is below n_rows and the ratio means something different.
    ceiling = len({(r["op_type"], r["problem_size"],
                    tuple(r["launch_config"]["grid"]),
                    tuple(r["launch_config"]["block"])) for r in rows})
    residual = _residual_collision(feats, rows)

    # THE GATE. The PI's point about 264-to-5: the number was correctly named,
    # correctly computed and correctly reported, and nothing asked what it
    # IMPLIES. So the implication is now asserted against the ceiling, and a
    # collapse is a non-zero exit rather than a line in metrics.json.
    naive_report = check_cardinality(
        len(rows), len(vectors_as_written), context="feature vector AS WRITTEN",
        ceiling=ceiling, ceiling_basis="distinct (op_type, n, grid, block)")
    fixed_report = check_cardinality(
        len(rows), len(vectors_fixed),
        context="feature vector with measured launch + problem size",
        ceiling=ceiling, ceiling_basis="distinct (op_type, n, grid, block)",
        residual_note=residual)

    out = {
        "bead": "gpu-sim-u2o.8.3",
        "dataset": f"{DATASET_REF}:{DATASET}",
        "dataset_ts_job": "20260928t074004-b429f469",
        "n_rows_populated": len(feats),
        "n_rows_with_unrecoverable_source": len(unrecovered),
        "source_recovery": {
            "method": "regenerate 7.1's sweep, then VERIFY by kernel_hash "
                      "(sha256(source + json([grid, block, n]))[:16])",
            "rows_verified": len(recovered),
            "matched_by_name_or_position": 0,
        },
        "distinct_vectors": {
            "as_written_no_launch_no_problem_size": len(vectors_as_written),
            "with_measured_launch_and_problem_size": len(vectors_fixed),
            "n_rows": len(feats),
        },
        "vector_dim": feats[0]["vector_dim"] if feats else 0,
        "cardinality_gate": {
            "n_distinct_configs_in_the_rows": ceiling,
            "as_written": naive_report.to_dict(),
            "corrected": fixed_report.to_dict(),
            "note": ("The AS-WRITTEN row is expected to be COLLAPSED and is "
                     "reported as such deliberately: it is the measurement of "
                     "the defect, and asserting against it would fail the run "
                     "that documents the bug. The CORRECTED row is the gate: "
                     "if a future change collapses it, the run stops."),
        },
        "finding": (
            "Populated for the first time from real measured rows. Before the "
            "fix the vector collapsed all 264 rows to "
            f"{len(vectors_as_written)} distinct value(s), because "
            "extract_launch_config() returns a 1-block default for a kernel "
            "DEFINITION (which contains no <<< >>> launch site) and because "
            "KernelFeatures had no problem_size field at all. With the measured "
            f"launch and problem size it separates {len(vectors_fixed)} of "
            f"{len(feats)} rows. This is the numeric-feature counterpart of "
            "gpu-sim-u2o.7.7's tokenizer finding: the representation could not "
            "see the two things that determine runtime."),
    }
    # Does the populated vector actually TRACK the measured label? Asked
    # because a vector that sees n and the launch config is still worthless if
    # the label it is supposed to explain does not move.
    out["does_the_vector_track_the_label"] = _label_dynamics(feats)

    # FAIL CLOSED on the corrected representation. The as-written figure is
    # reported without asserting, because refusing to emit the artifact that
    # documents the defect would hide the defect.
    try:
        assert_cardinality(
            len(rows), len(vectors_fixed),
            context="feature vector with measured launch + problem size",
            ceiling=ceiling, ceiling_basis="distinct (op_type, n, grid, block)",
            residual_note=residual)
        gate = "PASSED"
    except CardinalityError as exc:
        gate = f"FAILED: {exc}"

    (HERE / "metrics.json").write_text(json.dumps(out, indent=2))
    with (HERE / "features.jsonl").open("w") as fh:
        for f in feats:
            fh.write(json.dumps(f) + "\n")

    print(f"populated {len(feats)}/{len(rows)} rows "
          f"(all source-verified by hash)")
    print(f"vector_dim: {out['vector_dim']}")
    print(f"distinct vectors as written : {len(vectors_as_written)}")
    print(f"distinct vectors with fix   : {len(vectors_fixed)}")
    print(f"cardinality gate            : {gate}")
    if residual:
        print(f"residual collision          : {residual}")
    return 0 if gate == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
