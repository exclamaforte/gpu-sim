"""gpu-sim-u2o.6.6 TARGET driver (CPU-only): per-cell (median, sigma_log) + regime pair.

The deliverable is the TARGET for 9.1's learned model: a distribution per
cell, because 9.3's loss is Gaussian NLL on log-runtime and needs a mean AND
a log-variance. A scalar would throw away what 9.3 exists to consume.

Load-bearing rule (acceptance item 7): noise and model bias ride BESIDE each
other, never folded together. sigma_log comes from the cell's OWN ai_cv only
(never pooled); the 9.4 per-regime residuals are attached LABELLED as model
bias and are NOT subtracted -- target median == committed row median exactly
(gated below), so no model error could have been baked into the signal.

Regimes: measured_regime (on every row: launch_bound/measurable) AND the
roofline triple (9.4's construction, for bias keying) AND the tree binary
(latency/memory, for the disagree flag). Where measured and binary disagree
the cell is FLAGGED, not reassigned.

sigma_log = sqrt(ln(1 + ai_cv^2)), the exact lognormal map. The tiny-sample
correction is inside ai_cv itself: rows carry the SAMPLE cv
(statistics.stdev, ddof=1) over n=2 invocation medians -- Bessel, stated, no
further inflation invented. ai_cv == 0.0 maps to sigma_log == 0.0 exactly
(counted, not floored: a floor would be invented spread; 9.3 owns its eps).

Versioning: targets_version integer + recorded rule, in this artifact; row
protocols named in corpus_protocol.py's vocabulary (7.1 rows carry the
registered id; the e5q id is recorded PENDING registration -- adding it to
PROTOCOLS would trip 6.7.1's pinned rows==264 tests, a separate bead's
decision, so the gap is named here instead of hidden).
"""
import importlib.util
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "epic_u2o"))
import epic_u2o.roofline as roof_new  # noqa: E402
from epic_u2o.corpus_protocol import describe as describe_protocol  # noqa: E402

ZHQ = REPO / "experiments" / "gpu-sim-zhq" / "zhq_verify.py"
spec = importlib.util.spec_from_file_location("zhq_verify_mod", ZHQ)
zv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(zv)

P46 = REPO / "experiments" / "gpu-sim-46p" / "bw_verify.py"
spec46 = importlib.util.spec_from_file_location("bw_verify_mod", P46)
bwv = importlib.util.module_from_spec(spec46)
spec46.loader.exec_module(bwv)

FAILURES = []

TARGETS_VERSION = 1
CONSTRUCTION_RULE = (
    "v1: median = median of per-invocation medians (row median, unadjusted); "
    "sigma_log = sqrt(ln(1 + ai_cv^2)) from the cell's own sample ai_cv "
    "(statistics.stdev ddof=1 over n=2 invocation medians); regimes = "
    "measured_regime + tree binary + 9.4 triple; 9.4 residuals attached as "
    "labelled model bias, never subtracted.")

# Row provenance: file -> protocol id. The 7.1 id is registry-registered; the
# e5q id is recorded here PENDING registration (see module docstring).
PROTOCOL_BY_FILE = {
    "gpu-sim-u2o.7.1": "single_run_7_1",
    "gpu-sim-e5q": "single_run_e5q",
}
TS_BY_FILE = {
    "gpu-sim-u2o.7.1": "20260928t074004-b429f469",
    "gpu-sim-e5q": "20261010t224024-465ad815",
}

# 9.4's committed per-regime residuals (signed_bias_after), read live and
# exact-matched. These are MODEL bias: attached, never subtracted.
EXPECT_BIAS = {"latency->latency": -0.5638, "latency->memory": 0.378,
               "memory->memory": 1.1164}


def fail(msg):
    FAILURES.append(msg)
    print(f"TARGETS FAIL: {msg}", file=sys.stderr)


def cell_median(invocation_medians):
    """Item 9's estimator: median of per-invocation medians, never one run."""
    return statistics.median(invocation_medians)


def sigma_log_from_cv(ai_cv):
    """Per-cell log-uncertainty. Own cv only; cv==0 maps to 0.0 exactly."""
    if ai_cv is None or ai_cv < 0:
        fail(f"unusable ai_cv: {ai_cv!r}")
        return 0.0
    return math.sqrt(math.log1p(ai_cv * ai_cv))


def main():
    out_t = HERE / "targets.jsonl"
    out_m = HERE / "metrics.json"
    if "--metrics-out" in sys.argv:
        out_m = Path(sys.argv[sys.argv.index("--metrics-out") + 1])
    out_t_arg = None
    if "--targets-out" in sys.argv:
        out_t_arg = Path(sys.argv[sys.argv.index("--targets-out") + 1])

    rows = []
    for corpus, proto in PROTOCOL_BY_FILE.items():
        for line in (REPO / "experiments" / corpus / "measured.jsonl"
                     ).read_text().splitlines():
            r = json.loads(line)
            r["_protocol"] = proto
            rows.append(r)
    if len(rows) != 456:
        fail(f"population moved: {len(rows)} rows, expected 456")

    # Registry use: the 7.1 id must resolve; the e5q id must NOT (pending).
    try:
        describe_protocol("single_run_7_1")
    except Exception as e:  # noqa: BLE001
        fail(f"registry lost single_run_7_1: {e}")
    try:
        describe_protocol("single_run_e5q")
        fail("single_run_e5q unexpectedly registered; pending note is stale")
    except Exception:  # noqa: BLE001
        pass  # pending registration, as recorded

    # Contention rule (acceptance item 4): runs, reports, excludes nothing
    # because the corpus is clean -- a contended row would fail, not adjust.
    contended = [r["kernel_hash"] for r in rows
                 if r["queue_state"]["n_other_running"] != 0
                 or r["queue_state"]["n_other_active"] != 0]
    if contended:
        fail(f"contention: {len(contended)} rows ran contended: "
             f"{contended[:3]}")

    # 9.4 bias, live exact-match (the numbers that must not move silently).
    p46 = json.loads((REPO / "experiments/gpu-sim-46p/metrics.json"
                      ).read_text())["per_regime"]
    bias = {k: v["signed_bias_after"] for k, v in p46.items()}
    for k, v in EXPECT_BIAS.items():
        if bias.get(k) != v:
            fail(f"9.4 bias moved: {k} committed {bias.get(k)}, expected {v}")

    ancient = zv.load_old_roofline()
    old = bwv.load_before()
    targets = []
    reg_cv = defaultdict(list)
    meas_cv = defaultdict(list)
    for r in rows:
        for k in ("ai_cv", "n_invocations", "invocation_medians_ms",
                  "median_runtime_ms", "measured_regime"):
            if r.get(k) is None:
                fail(f"row {r.get('kernel_hash')} missing {k}")
        meds = r["invocation_medians_ms"]
        med = cell_median(meds)
        if med != r["median_runtime_ms"]:
            fail(f"estimator drift on {r['kernel_hash']}: median-of-medians "
                 f"{med} != committed {r['median_runtime_ms']}")
        if r["n_invocations"] != len(meds) or r["n_invocations"] < 2:
            fail(f"invocation count broken on {r['kernel_hash']}: "
                 f"n={r['n_invocations']} medians={len(meds)}")
        f = zv.row_features(r)
        rz = ancient.roofline_predict(zv.to_old(ancient, f))
        rb = old.roofline_predict(zv.to_old(old, f))
        triple = f"{rz.bottleneck}->{rb.bottleneck}"
        binary = roof_new.roofline_predict(zv.to_new(f)).bottleneck
        if binary not in ("latency", "memory"):
            fail(f"non-binary tree bottleneck on {r['kernel_hash']}: {binary}")
        # Disagreement mapping, stated: launch_bound ~ latency,
        # measurable ~ memory. Flag where they part.
        agree = (r["measured_regime"] == "launch_bound") == (binary == "latency")
        sig = sigma_log_from_cv(r["ai_cv"])
        if not math.isfinite(sig) or sig < 0:
            fail(f"non-finite sigma on {r['kernel_hash']}: {sig}")
        targets.append({
            "kernel_hash": r["kernel_hash"],
            "op_type": r["op_type"],
            "protocol": r["_protocol"],
            "targets_version": TARGETS_VERSION,
            "median_runtime_ms": med,  # == row median: bias NOT subtracted
            "sigma_log": sig,
            "n_invocation": r["n_invocations"],
            "ai_cv": r["ai_cv"],
            "measured_regime": r["measured_regime"],
            "roofline_regime": binary,
            "roofline_triple": triple,
            "model_bias": bias.get(triple),  # labelled bias, beside noise
            "regime_disagree": not agree,
            "deliberately_poor": bool(r["deliberately_poor"]),
        })
        reg_cv[triple].append(r["ai_cv"])
        meas_cv[r["measured_regime"]].append(r["ai_cv"])

    # Class censuses (constructions must not drift silently).
    if sorted(reg_cv) != ["latency->latency", "latency->memory",
                          "memory->memory"]:
        fail(f"triple classes moved: {sorted(reg_cv)}")
    triple_n = {k: len(v) for k, v in reg_cv.items()}
    if triple_n != {"latency->latency": 220, "latency->memory": 204,
                    "memory->memory": 32}:
        fail(f"triple census moved: {triple_n}")
    binary_n = defaultdict(int)
    for t in targets:
        binary_n[t["roofline_regime"]] += 1
    if dict(binary_n) != {"latency": 240, "memory": 216}:
        fail(f"binary census moved: {dict(binary_n)}")
    flags = [t for t in targets if t["regime_disagree"]]
    if len(flags) != 43 or any(f["measured_regime"] != "launch_bound"
                               or f["roofline_regime"] != "memory"
                               for f in flags):
        fail(f"disagreement set moved: {len(flags)} flags, "
             f"one-sided launch_bound x memory expected")

    def noise_table(groups):
        return {k: {"n": len(v),
                    "median_ai_cv": statistics.median(v),
                    "min_ai_cv": min(v),
                    "max_ai_cv": max(v),
                    "median_sigma_log": statistics.median(
                        [sigma_log_from_cv(x) for x in v])}
                for k, v in sorted(groups.items())}

    zeros = sum(1 for t in targets if t["sigma_log"] == 0.0)
    metrics = {
        "bead": "gpu-sim-u2o.6.6",
        "targets_version": TARGETS_VERSION,
        "construction_rule": CONSTRUCTION_RULE,
        "n_targets": len(targets),
        "protocols_used": [
            {"id": "single_run_7_1", "rows": 264,
             "ts_job": TS_BY_FILE["gpu-sim-u2o.7.1"],
             "registry": "registered",
             "pooling_decision": "combined with single_run_e5q rows for "
             "per-regime NOISE DESCRIPTION only; no cell's sigma borrows "
             "another cell's noise, within or across protocols."},
            {"id": "single_run_e5q", "rows": 192,
             "ts_job": TS_BY_FILE["gpu-sim-e5q"],
             "registry": "PENDING registration (would trip 6.7.1's pinned "
             "rows==264 tests; separate bead's decision)",
             "pooling_decision": "same harness constants (2 inv x 20 trials, "
             "identical ai_cv construction); see above."},
        ],
        "estimator": ("median of per-invocation medians == committed row "
                      "median on all 456 rows; n_invocation attached per row; "
                      "never a single-run figure."),
        "sigma_construction": (
            "sigma_log = sqrt(ln(1 + ai_cv^2)) from the cell's own sample "
            "ai_cv (statistics.stdev, ddof=1, over n=2 invocation medians -- "
            "Bessel IS the tiny-sample correction; no further inflation). "
            "ai_cv == 0.0 -> sigma_log == 0.0 exactly (no invented floor; "
            "9.3 owns its loss eps)."),
        "noise_by_roofline_triple": noise_table(reg_cv),
        "noise_by_measured_regime": noise_table(meas_cv),
        "noise_floor_crosscheck_661": {
            "661_characterisation": ("across-invocation CV ~7.4% sub-10us, "
                                     "~0.13% over-10us (anchor cells, 3 inv, "
                                     "200 trials)"),
            "corpus_sub_10us": {"n": 283, "median_ai_cv": statistics.median(
                [r["ai_cv"] for r in rows
                 if r["median_runtime_ms"] * 1000 < 10])},
            "corpus_over_10us": {"n": 173, "median_ai_cv": statistics.median(
                [r["ai_cv"] for r in rows
                 if r["median_runtime_ms"] * 1000 >= 10])},
            "note": ("same direction, noisier estimate: corpus runs 2 inv x "
                     "20 trials vs 6.6.1's 3 x 200. Populations differ; the "
                     "numbers are beside each other, not pooled.")},
        "model_bias_beside_noise": {
            "values": bias,
            "not_subtracted": ("target median == committed row median on all "
                               "456 rows (gated); subtracting a model's error "
                               "into the supervised signal would train 9.1 to "
                               "reproduce the roofline's mistakes."),
            "candidates_pointer": ("9.4 REPORT.md Residual A: cache-resident "
                                   "traffic (primary); Residual B: n-local "
                                   "floor or above-floor per-launch cost.")},
        "regime_disagreement": {
            "mapping": ("agree iff (measured==launch_bound) == "
                        "(roofline==latency)"),
            "n_flags": len(flags),
            "shape": "one-sided: all 43 are launch_bound x memory (model "
                     "says memory-bound, measurement sits at the floor) -- "
                     "information about the model, not noise.",
            "flagged_hashes": sorted(f["kernel_hash"] for f in flags)},
        "contention": {"n_contended": len(contended), "n_excluded": 0,
                       "verdict": "456/456 cells queue_state (0,0); the "
                                  "rule runs, nothing adjusted."},
        "conventions": ("population 456 (264 ref + 192 additive); byte "
                        "convention 8.3 arity (subscripts + *p + atomics + "
                        "float scalars, x4B, x covers; loop-resident once); "
                        "estimator median-of-invocation-medians; 8 ops x 1 "
                        "kernel each; deliberately_poor carried on 8 rows, "
                        "not excluded (no rule authored; 9.1's sampling "
                        "decision)."),
        "zero_sigma_cells": zeros,
        "verdict": "pass" if not FAILURES else "fail",
        "failures": FAILURES}

    # Item 6 gate: latency noise materially exceeds memory noise, or say so.
    lat = metrics["noise_by_roofline_triple"]["latency->latency"]["median_ai_cv"]
    mem = metrics["noise_by_roofline_triple"]["memory->memory"]["median_ai_cv"]
    if not lat > 5 * mem:
        fail(f"latency noise does not dwarf memory noise: {lat} vs {mem}")

    m_text = json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    t_text = "".join(json.dumps(t, sort_keys=True) + "\n" for t in targets)
    if out_t_arg is not None:
        out_t_arg.write_text(t_text)
        out_m.write_text(m_text)
    else:
        out_t.write_text(t_text)
        out_m.write_text(m_text)
    print(f"n={len(targets)} flags={len(flags)} zeros={zeros} "
          f"verdict={metrics['verdict']}", flush=True)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
