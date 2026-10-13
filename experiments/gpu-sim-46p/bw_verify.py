"""gpu-sim-46p (CPU-only): memory arm flat vs occupancy-derated, per regime.

Compares pinned pre-46p roofline (e0016a7, the 8.3 merge) against the tree,
over the 456-row corpus, split by bottleneck TRANSITION class -- the three
regimes the corpus now separates into:
  latency->latency  (latency-bound throughout)
  latency->memory   (memory-bound via the latency map: the arm came alive)
  memory->memory    (memory-bound throughout)

PRE-REGISTERED EXPECTATION (acceptance 3 -- written BEFORE the re-run, and
a direction, not a number; picking the form after seeing the error move
would be fitting):
  form (a): achieved bandwidth flat, occupancy kept on the compute arm only.
  Memory times shrink by ~1/occupancy (mean occ ~0.68, so ~x0.68). The +0.37
  signed bias should fall toward zero or slightly negative; mean abs should
  drop substantially from ~1.0. Stated cost: rows too small to saturate DRAM
  may now UNDERpredict on the memory arm -- ceiling semantics, the latency
  floor binds them instead.

No per-op or per-regime constants anywhere in this script. No GPU.
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

# Pre-46p reference: the 8.3 merge. Merged and immutable (same pin pattern
# as g8x/zhq: a fixed hash, both-direction assertions).
PRE_REF = "e0016a7232677301845353ca20a9dc246b0bdd1f"

EXPECTATION = (
    "form (a): achieved 1688.592 GB/s flat on the memory arm, occupancy "
    "kept on the compute arm only. Memory times shrink ~x0.68; the +0.37 "
    "signed bias is expected to fall toward zero-or-negative and mean abs "
    "to drop substantially from ~1.0. Stated cost: sub-saturation rows may "
    "underpredict on the memory arm (ceiling semantics).")


def load_before():
    src = subprocess.run(["git", "show", f"{PRE_REF}:epic_u2o/roofline.py"],
                         cwd=str(REPO), capture_output=True, text=True,
                         check=True).stdout
    assert "specs.memory_bandwidth_gbps * occupancy" in src, \
        "pre-46p derate marker gone: wrong ref?"
    assert "THEORETICAL_READ_BW_GBS" not in src, \
        "46p already present in pre-46p ref?!"
    spec = importlib.util.spec_from_loader("roofline_before46p", loader=None)
    mod = importlib.util.module_from_spec(spec)
    exec(compile(src, "roofline_before46p.py", "exec"), mod.__dict__)
    return mod


def main():
    out = Path(sys.argv[sys.argv.index("--metrics-out") + 1]) \
        if "--metrics-out" in sys.argv else None
    old = load_before()
    # Regime CLASS comes from the 8.3 map (acceptance 2 names these three):
    # zhq-pinned pre-zhq roofline (invented latency arm) -> pre-46p roofline
    # (derated memory arm), both fed the LIVE 8.3 features. That reproduces
    # the 220/204/32 split. The reported before/after within each class is
    # the 46p change (derated -> flat). A second key records the 46p-internal
    # transition, so rows the flat arm drops back to the floor stay visible.
    ancient = zv.load_old_roofline()
    rows = []
    for c in ("experiments/gpu-sim-u2o.7.1/measured.jsonl",
              "experiments/gpu-sim-e5q/measured.jsonl"):
        rows += [json.loads(l) for l in (REPO / c).read_text().splitlines()]

    reg = defaultdict(lambda: [[], [], [], []])  # abs_b, abs_a, bias_b, bias_a
    trans46 = defaultdict(int)
    tot_b, tot_a, bias_b, bias_a = [], [], [], []
    for r in rows:
        f = zv.row_features(r)
        actual = r["median_runtime_ms"]
        if actual <= 0:
            continue
        rz = ancient.roofline_predict(zv.to_old(ancient, f))
        rb = old.roofline_predict(zv.to_old(old, f))
        ra = roof_new.roofline_predict(zv.to_new(f))
        cls = f"{rz.bottleneck}->{rb.bottleneck}"
        trans46[f"{rb.bottleneck}->{ra.bottleneck}"] += 1
        eb, ea = ((rb.predicted_time_ms - actual) / actual,
                  (ra.predicted_time_ms - actual) / actual)
        reg[cls][0].append(abs(eb))
        reg[cls][1].append(abs(ea))
        reg[cls][2].append(eb)
        reg[cls][3].append(ea)
        tot_b.append(abs(eb))
        tot_a.append(abs(ea))
        bias_b.append(eb)
        bias_a.append(ea)
    regimes = {k: {"n": len(v[0]),
                   "mean_abs_before": round(statistics.fmean(v[0]), 4),
                   "mean_abs_after": round(statistics.fmean(v[1]), 4),
                   "signed_bias_before": round(statistics.fmean(v[2]), 4),
                   "signed_bias_after": round(statistics.fmean(v[3]), 4)}
               for k, v in sorted(reg.items())}
    metrics = {
        "bead": "gpu-sim-46p", "n_rows": len(tot_b),
        "pre_ref": PRE_REF,
        "preregistered_expectation": EXPECTATION,
        "overall_mean_abs_before": round(statistics.fmean(tot_b), 4),
        "overall_mean_abs_after": round(statistics.fmean(tot_a), 4),
        "signed_bias_before": round(statistics.fmean(bias_b), 4),
        "signed_bias_after": round(statistics.fmean(bias_a), 4),
        "per_regime": regimes,
        "transition_46p_internal": dict(sorted(trans46.items())),
        "verdict": "pass"}
    if out:
        out.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(f"n={len(tot_b)} abs {metrics['overall_mean_abs_before']} -> "
          f"{metrics['overall_mean_abs_after']} bias "
          f"{metrics['signed_bias_before']} -> {metrics['signed_bias_after']}",
          flush=True)
    for k, v in sorted(regimes.items()):
        print(f"  {k}: n={v['n']} abs {v['mean_abs_before']} -> "
              f"{v['mean_abs_after']} bias {v['signed_bias_before']} -> "
              f"{v['signed_bias_after']}", flush=True)
    print(f"verdict={metrics['verdict']} -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
