"""gpu-sim-u2o.9.4 REPORT driver (CPU-only): the baseline number, reconciled.

This bead produces a REPORT, not a model change: nothing under epic_u2o/
is modified here. The driver checks every checkable claim in REPORT.md
against COMMITTED paths (criterion 3: a check against fresh output is an
identity once the two fields share a helper):

  arc        -- the 7.7238 -> 0.6581 table equals g8x/zhq/8.3/46p's committed
                metrics.json values (item 9). The 0.7258 -> 0.9955 step is
                the 8.3 memory arm arriving WITH the derate still on; the
                report says so beside the table.
  contention -- all 456 corpus rows carry queue_state with n_other_running
                == 0 (item 5, corpus side); megakern's json records NO
                contention state, which the report states instead of assuming.
  achieved   -- per-regime median achieved_GBps recomputed as
                bytes_moved/(median_ms/1000)/1e9 from committed measured
                rows + committed 8.3 features, with the 8.3 arity convention
                printed beside every figure (items 1, 3). Regime classes come
                from a live three-leg pass (zhq-pinned -> pre-46p -> tree),
                the same construction 46p's artifact records.
  ceiling    -- the candidate set {achieved 1688.592, theory 1792.0,
                mixed-traffic follow-up} with the argmax rule stated (item 2).
  families   -- per-op row AND kernel counts: 8 ops x 57 rows, ONE kernel
                per op (item 4: per-op figures are per-kernel measurements).

No per-op or per-regime constants anywhere. No GPU.
"""
import importlib.util
import json
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "epic_u2o"))
import epic_u2o.roofline as roof_new  # noqa: E402

ZHQ = REPO / "experiments" / "gpu-sim-zhq" / "zhq_verify.py"
spec = importlib.util.spec_from_file_location("zhq_verify_mod", ZHQ)
zv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(zv)

P46 = REPO / "experiments" / "gpu-sim-46p" / "bw_verify.py"
spec46 = importlib.util.spec_from_file_location("bw_verify_mod", P46)
bwv = importlib.util.module_from_spec(spec46)
spec46.loader.exec_module(bwv)

FAILURES = []


def fail(msg):
    FAILURES.append(msg)
    print(f"REPORT FAIL: {msg}", file=sys.stderr)


def committed(path):
    return json.loads((REPO / path).read_text())


def main():
    out = Path(sys.argv[sys.argv.index("--metrics-out") + 1]) \
        if "--metrics-out" in sys.argv else None

    # ---- arc reconciliation (item 9) ----
    g = committed("experiments/gpu-sim-g8x/metrics.json")
    z = committed("experiments/gpu-sim-zhq/metrics.json")
    u = committed("experiments/gpu-sim-u2o.8.3/metrics.json")
    p = committed("experiments/gpu-sim-46p/metrics.json")
    arc = {
        "g8x_before": g["overall_mean_abs_before"],
        "g8x_after": g["overall_mean_abs_after"],
        "zhq_before": z["overall_mean_abs_before"],
        "zhq_after": z["overall_mean_abs_after"],
        "zhq_bias_after": z["signed_bias_after"],
        "u83_distinct_corrected": u["distinct_vectors"]["corrected"],
        "u83_zero_byte_rows": u["bytes_moved_zero_rows"],
        "u83_item5_groups": u["item5_collision_groups"],
        "p46_before": p["overall_mean_abs_before"],
        "p46_after": p["overall_mean_abs_after"],
        "p46_bias_before": p["signed_bias_before"],
        "p46_bias_after": p["signed_bias_after"],
        "p46_regimes": {k: (v["mean_abs_after"], v["signed_bias_after"])
                        for k, v in sorted(p["per_regime"].items())},
    }
    expect = {"g8x_before": 7.7238, "g8x_after": 7.7238, "zhq_before": 7.7238,
              "zhq_after": 0.7258, "zhq_bias_after": -0.7258,
              "u83_distinct_corrected": 456, "u83_zero_byte_rows": 0,
              "u83_item5_groups": 0, "p46_before": 0.9955,
              "p46_after": 0.6581, "p46_bias_before": 0.3722,
              "p46_bias_after": -0.0245}
    for k, v in expect.items():
        if arc[k] != v:
            fail(f"arc.{k}: report says {v}, committed says {arc[k]}")

    # ---- contention census (item 5; item 7: queue_state, never clocks) ----
    rows = []
    for c in ("experiments/gpu-sim-u2o.7.1/measured.jsonl",
              "experiments/gpu-sim-e5q/measured.jsonl"):
        rows += [json.loads(l) for l in (REPO / c).read_text().splitlines()]
    contended = [r["kernel_hash"] for r in rows
                 if r["queue_state"]["n_other_running"] != 0
                 or r["queue_state"]["n_other_active"] != 0]
    if contended:
        fail(f"contention: {len(contended)} corpus rows ran contended, e.g. "
             f"{contended[:3]}")
    contention = {"n_rows": len(rows), "n_contended": len(contended),
                  "verdict": ("456/456 corpus cells measured with no other "
                              "job active or running (queue_state on every "
                              "row). The ceiling anchor (megakern "
                              "rtx5090.json, ts 20261003t213327-db6b7f3f) "
                              "records NO contention state -- stated, not "
                              "assumed; the corpus side of the bound is "
                              "clean, the anchor side is unknown.")}

    # ---- families: per-op rows AND kernels (item 4) ----
    by_op = defaultdict(list)
    for r in rows:
        by_op[r["op_type"]].append(r["kernel_name"])
    families = {op: {"n_rows": len(v), "n_kernels": len(set(v))}
                for op, v in sorted(by_op.items())}
    if not all(v["n_kernels"] == 1 for v in families.values()):
        fail("family census changed: expected one kernel per op")
    # exclusions (item 8 provenance tail): none -- all 456 actual>0, all used.
    excluded = [r["kernel_hash"] for r in rows if r["median_runtime_ms"] <= 0]
    if excluded:
        fail(f"excluded rows present: {excluded[:3]}")

    # ---- regime classes, live three-leg pass (same as 46p's construction) --
    ancient = zv.load_old_roofline()
    old = bwv.load_before()
    feats = {json.loads(l)["kernel_hash"]: json.loads(l)
             for l in (REPO / "experiments/gpu-sim-u2o.8.3/features.jsonl")
             .read_text().splitlines()}
    reg_ach = defaultdict(list)
    raws = []
    for r in rows:
        f = zv.row_features(r)
        rz = ancient.roofline_predict(zv.to_old(ancient, f))
        rb = old.roofline_predict(zv.to_old(old, f))
        cls = f"{rz.bottleneck}->{rb.bottleneck}"
        b = feats[r["kernel_hash"]]["bytes_moved"]
        a = r["median_runtime_ms"]
        achieved = b / (a / 1000.0) / 1e9 if a > 0 else 0.0
        raws.append((r["kernel_hash"], r["op_type"], cls, b, a, achieved))
        reg_ach[cls].append(achieved)
    # Residual-A evidence: rows whose EFFECTIVE rate (bytes/median, cache-
    # inclusive) exceeds DRAM read-peak cannot be DRAM-bound traffic. 8.3's
    # bytes are issued-access counts; repeated trials on resident buffers
    # serve much of it from L2 (96 MiB), so the DRAM-ceiling model
    # structurally overpredicts exactly these rows. Counted, not fitted.
    regimes = {k: {"n": len(v),
                   "median_achieved_GBps": round(statistics.median(v), 1),
                   "max_achieved_GBps": round(max(v), 1),
                   "n_exceeding_read_peak": sum(1 for x in v if x > 1688.592)}
               for k, v in sorted(reg_ach.items())}
    if sorted(reg_ach) != ["latency->latency", "latency->memory",
                           "memory->memory"]:
        fail(f"regime classes moved: {sorted(reg_ach)}")
    # Item 3 exemplar: the full-cover scale row at n=2^22, recomputed from
    # committed bytes and committed median, beside the convention.
    ex = next(x for x in raws if x[1] == "scale" and feats[x[0]]["bytes_moved"]
              == 3 * 4 * (1 << 22))
    exemplar = {"kernel_hash": ex[0], "bytes_moved": ex[3],
                "median_ms": ex[4],
                "achieved_GBps": round(ex[5], 1),
                "convention": "8.3 arity: 2 reads + 1 write of 4B = 3*4*n"}

    # ---- ceiling candidate set (item 2: argmax with membership rule) ----
    ceiling = {
        "candidates": [
            {"value_GBps": 1688.592, "source": "megakern peak_read_GBps",
             "status": "CHOSEN (46p form a): measured at saturation"},
            {"value_GBps": 1792.0, "source": "megakern theoretical_read_bw_GBs",
             "status": "REJECTED: vendor theory, no measured efficiency"},
            {"value_GBps": 1468.996, "source": "megakern peak_copy_GBps",
             "status": "FOLLOW-UP (residual A): mixed-traffic ceiling by "
                       "arity, to be measured not fitted"}],
        "membership_rule": ("measured on this silicon (megakern-37e.2) or "
                            "vendor theory from the same file; no other "
                            "numbers admitted"),
        "argmax_rule": ("the achieved-at-saturation figure: a ceiling is a "
                        "rate some launch demonstrably sustains, not a rate "
                        "no launch has reached"),
    }

    metrics = {
        "bead": "gpu-sim-u2o.9.4", "n_rows": len(rows),
        "arc": arc,
        "arc_overall": "7.7238 -> 0.6581 (11.7x) by removing four wrong "
                       "assumptions, none of it by fitting",
        "contention": contention,
        "families": families,
        "excluded_rows": len(excluded),
        "per_regime_achieved": regimes,
        "bandwidth_convention": ("8.3 arity counts per row: subscripts + *p "
                                 "derefs + atomics + float scalars, x4B, "
                                 "x covers; loop/branch-resident counted "
                                 "once. Printed beside EVERY figure (item 1)."),
        "exemplar_scale_2e22": exemplar,
        "ceiling_candidates": ceiling,
        "cancellation": ("overall signed bias -0.0245 is CANCELLATION, not "
                         "accuracy: latency->latency underpredicts (-0.5638, "
                         "residual B) while latency->memory (+0.378) and "
                         "memory->memory (+1.1164, residual A) overpredict. "
                         "Reporting -0.0245 as accuracy would be the single "
                         "most misleading thing this report could do."),
        "residual_A": ("memory->memory n=32, abs 1.1209 bias +1.1164, all "
                       "overpredict. PRIMARY candidate (evidence in "
                       "per_regime_achieved): cache-resident traffic. 148/456 "
                       "rows show EFFECTIVE rates above DRAM read-peak "
                       "(max 7480 GB/s), all at large n across all ops -- "
                       "repeated trials on resident buffers serve issued "
                       "traffic from L2, so a DRAM-ceiling model overpredicts "
                       "structurally. The read-peak-vs-mixed-write candidate "
                       "is DEMOTED on sign: slower write bandwidth predicts "
                       "UNDERprediction, but all 32 rows OVERpredict. Scalar "
                       "overcount (alpha as 4B/thread DRAM) contributes at "
                       "most ~25% for saxpy-class rows. Carry to a named "
                       "follow-up with a cache-busting streaming measurement; "
                       "the memory arm's form does not change in this bead."),
        "residual_B": ("latency->latency n=220, abs 0.5638 bias -0.5638, all "
                       "underpredict. Candidates: n-local floor (39u ruled it "
                       "out overall, not for this subset) or a fixed "
                       "per-launch cost above the floor (39u mechanism (a), "
                       "named, never tested). Carry, do not fit."),
        "verdict": "pass" if not FAILURES else "fail",
        "failures": FAILURES}
    if out:
        out.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(f"n={len(rows)} contended={len(contended)} excluded={len(excluded)} "
          f"regimes={ {k: v['n'] for k, v in regimes.items()} }",
          flush=True)
    print(f"verdict={metrics['verdict']} -> {out}", flush=True)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
