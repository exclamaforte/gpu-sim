#!/usr/bin/env python3
"""Populate the numeric feature vector from all 456 REAL measured rows. 8.3.

Join: regenerate each corpus from its own declared sweep, VERIFY every row
by kernel_hash (sha256(source + json([grid, block, n]))[:16] -- the unified
kernel_hash_for), then extract_features with the MEASURED launch config and
the MEASURED problem_size. A row whose hash cannot be recovered fails the
run; nothing is matched by name or position.

Gates (nonzero exit on failure):
  item 4 -- 456/456 rows joined with reads/writes actually COMPUTED
            (bytes_moved == 0 anywhere fails: the letter-without-spirit run).
  item 5 -- no two rows with DIFFERENT median_runtime_ms share a vector.
  item 9 -- scale at n=2^22 full-cover: 2 reads + 1 write of 4B = 3*4*n,
            checked against the INDEPENDENT hand computation, never the error.

No GPU, no ts job: reads committed artifacts, re-derives from the
deterministic generator.
"""
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from epic_u2o.features import LaunchConfig, extract_features  # noqa: E402
from epic_u2o.generator_stage1 import (  # noqa: E402
    generate_kernels, kernel_hash_for)
from epic_u2o.representation_cardinality import (  # noqa: E402
    CardinalityError, assert_cardinality, check_cardinality)

CORPORA = {
    "u2o.7.1": {"path": "experiments/gpu-sim-u2o.7.1/measured.jsonl",
                "sizes": [1 << 12, 1 << 14, 1 << 16, 1 << 18],
                "ts_job": "20260928t074004-b429f469"},
    "e5q": {"path": "experiments/gpu-sim-e5q/measured.jsonl",
            "sizes": [1 << 22, 1 << 23, 1 << 24],
            "ts_job": "20261010t224024-465ad815"},
}
BLOCKS = [[64, 1, 1], [128, 1, 1], [256, 1, 1], [1024, 1, 1]]
OPS = ["scale", "add", "saxpy", "relu", "copy", "triad", "mul", "div"]

FAILURES = []


def fail(msg):
    FAILURES.append(msg)
    print(f"GATE FAIL: {msg}", file=sys.stderr)


def grids_for(n, block):
    """Both corpora's declared rule: full + 1/2 + 1/8 of full, 1-block poor."""
    bt = block[0] * block[1] * block[2]
    full = max(1, n // bt)
    out = [[full, 1, 1], [max(1, full // 2), 1, 1], [max(1, full // 8), 1, 1]]
    if full > 64:
        out.append([1, 1, 1])
    return out


def regenerate(sizes):
    out = {}
    for n in sizes:
        for block in BLOCKS:
            for grid in grids_for(n, block):
                for spec in generate_kernels(
                        ops=OPS, grid_sweeps=[grid], block_sweeps=[block],
                        problem_sizes=[n]):
                    d = spec.to_dict()
                    h = kernel_hash_for(d["source"], d["grid"], d["block"], n)
                    assert h == hashlib.sha256(
                        (d["source"] + json.dumps(
                            [d["grid"], d["block"], n])).encode()
                    ).hexdigest()[:16], "unified hash disagrees with inline"
                    out[h] = d
    return out


def main() -> int:
    rows = []
    for name, c in CORPORA.items():
        p = REPO / c["path"]
        if not p.exists():
            fail(f"item 4: corpus file missing: {c['path']}")
            return 1
        for line in p.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                r["_corpus"] = name
                rows.append(r)

    sources = {}
    for name, c in CORPORA.items():
        sources.update(regenerate(c["sizes"]))

    unrecovered = [r for r in rows if r["kernel_hash"] not in sources]
    if unrecovered:
        fail(f"item 4: {len(unrecovered)}/{len(rows)} rows unrecoverable by "
             f"hash, e.g. {unrecovered[0]['kernel_hash']} "
             f"({unrecovered[0]['_corpus']})")
        return 1

    disjoint = len({r["kernel_hash"] for r in rows}) == len(rows)
    if not disjoint:
        fail("corpora overlap: joined set is not 456 distinct hashes")

    feats = []
    for r in rows:
        src = sources[r["kernel_hash"]]["source"]
        lc = r["launch_config"]
        measured_lc = LaunchConfig(
            grid_x=lc["grid"][0], grid_y=lc["grid"][1], grid_z=lc["grid"][2],
            block_x=lc["block"][0], block_y=lc["block"][1],
            block_z=lc["block"][2],
            shared_mem_bytes=lc.get("shared_mem_bytes", 0))
        f = extract_features(src, launch_config=measured_lc,
                             problem_size=r["problem_size"])
        d = f.to_dict()
        if d["covers"] != r["covers"]:
            fail(f"covers mismatch {r['kernel_hash']}: vector {d['covers']} "
                 f"!= row {r['covers']}")
        feats.append({
            "kernel_hash": r["kernel_hash"], "corpus": r["_corpus"],
            "op_type": r["op_type"], "problem_size": r["problem_size"],
            "covers": d["covers"],
            "measured_median_ms": r["median_runtime_ms"],
            "measured_regime": r.get("measured_regime"),
            "vector": f.to_vector(), "vector_dim": f.vector_dim,
            "reads": d["estimated_global_reads"],
            "writes": d["estimated_global_writes"],
            "bytes_moved": d["estimated_bytes_moved"],
            "flops": d["estimated_flops"],
        })

    # Item 4, spirit: bytes_moved == 0 anywhere means the D5 fix did not land.
    zeroed = [f["kernel_hash"] for f in feats if f["bytes_moved"] == 0]
    if zeroed:
        fail(f"item 4: {len(zeroed)} rows with bytes_moved still 0, e.g. "
             f"{zeroed[:3]}")

    # Item 5, THE gate: same vector + different runtime = fail, named groups.
    groups = defaultdict(list)
    for f in feats:
        groups[tuple(f["vector"])].append(f)
    collisions = {v: g for v, g in groups.items()
                  if len({x["measured_median_ms"] for x in g}) > 1}
    if collisions:
        worst = max(collisions.values(),
                    key=lambda g: max(x["measured_median_ms"] for x in g)
                    / min(x["measured_median_ms"] for x in g) - 1)
        ts = sorted(x["measured_median_ms"] for x in worst)
        fail(f"item 5: {len(collisions)} same-vector groups with different "
             f"runtimes; worst spread {(max(ts) / min(ts) - 1) * 100:.1f}% "
             f"over {[(x['corpus'], x['op_type'], x['problem_size']) for x in worst][:6]}")

    # Item 9: the independent hand computation. scale at n=2^22, full cover
    # (covers == n so the guard changes nothing): 2 reads + 1 write of 4B.
    hand = [f for f in feats if f["op_type"] == "scale"
            and f["problem_size"] == 1 << 22 and f["covers"] == 1 << 22]
    if not hand:
        fail("item 9: no full-cover scale row at n=2^22 to check against")
    else:
        h = hand[0]
        n = 1 << 22
        if not (h["reads"] == 2 * n and h["writes"] == 1 * n
                and h["bytes_moved"] == 3 * 4 * n):
            fail(f"item 9: scale n=2^22 reads={h['reads']} writes={h['writes']} "
                 f"bytes={h['bytes_moved']}, expected {2 * n}/{1 * n}/{3 * 4 * n}")

    ceiling = len({(r["op_type"], r["problem_size"],
                    tuple(r["launch_config"]["grid"]),
                    tuple(r["launch_config"]["block"])) for r in rows})
    naive_vectors = set()
    for r in rows:
        src = sources[r["kernel_hash"]]["source"]
        naive_vectors.add(tuple(round(x, 6)
                                for x in extract_features(src).to_vector()))
    naive_report = check_cardinality(
        len(rows), len(naive_vectors), context="feature vector AS WRITTEN",
        ceiling=ceiling, ceiling_basis="distinct (op_type, n, grid, block)")
    fixed_report = check_cardinality(
        len(rows), len(groups),
        context="feature vector corrected (launch + size + covers + r/w)",
        ceiling=ceiling, ceiling_basis="distinct (op_type, n, grid, block)")

    out = {
        "bead": "gpu-sim-u2o.8.3",
        "corpora": {name: {"path": c["path"], "ts_job": c["ts_job"],
                           "n_rows": sum(1 for r in rows if r["_corpus"] == name)}
                   for name, c in CORPORA.items()},
        "n_rows_populated": len(feats),
        "n_rows_with_unrecoverable_source": 0,
        "hash_sets_disjoint": disjoint,
        "source_recovery": {
            "method": "regenerate each corpus sweep, VERIFY by kernel_hash "
                      "(unified kernel_hash_for); 0 matched by name/position",
            "rows_verified": len(rows),
        },
        "distinct_vectors": {
            "as_written_no_launch_no_size": len(naive_vectors),
            "corrected": len(groups),
            "n_rows": len(feats),
        },
        "vector_dim": feats[0]["vector_dim"] if feats else 0,
        "bytes_moved_zero_rows": len(zeroed),
        "item5_collision_groups": len(collisions),
        "item9_hand_check": {
            "op": "scale", "n": 1 << 22,
            "rule": "full-cover row: 2 reads + 1 write of 4B = 3*4*n",
            "observed": ({k: hand[0][k] for k in
                          ("reads", "writes", "bytes_moved")}
                         if hand else None),
        },
        "cardinality_gate": {
            "n_distinct_configs_in_the_rows": ceiling,
            "as_written": naive_report.to_dict(),
            "corrected": fixed_report.to_dict(),
        },
        "features_added_vs_original_27": {
            "problem_size": "vector[0]: the size the launch covers; without "
                            "it same-kernel different-n launches collide",
            "covers": "min(threads, n): the work actually performed; also "
                      "the scale for all per-thread estimates (guarded-out "
                      "threads do no memory/compute)",
            "estimated_add/mul/div_ops": "operator split of the flop count: "
                      "add/mul/div run on different units (div 137% slower "
                      "at the same launch), one flop number cannot see that",
            "num_call_sites": "opaque work the op counters cannot see "
                      "(fmaxf in relu, atomicAdd in stage-5, __syncthreads)",
            "note": "positions 0..27 unchanged; the five new fields append "
                    "at 28..32. normalize rewritten by NAME after the old "
                    "positional code indexed grid_x at a problem_size slot.",
        },
    }
    (HERE / "metrics.json").write_text(json.dumps(out, indent=2))
    with (HERE / "features.jsonl").open("w") as fh:
        for f in feats:
            fh.write(json.dumps(f) + "\n")

    print(f"populated {len(feats)}/{len(rows)} rows (hash-verified, "
          f"disjoint={disjoint})")
    print(f"vector_dim: {out['vector_dim']}")
    print(f"distinct as-written: {len(naive_vectors)}, corrected: {len(groups)}")
    print(f"bytes_moved==0 rows: {len(zeroed)}; item-5 groups: {len(collisions)}")
    if FAILURES:
        print(f"VERDICT: FAIL ({len(FAILURES)} gate failures)")
        return 1
    print("VERDICT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
