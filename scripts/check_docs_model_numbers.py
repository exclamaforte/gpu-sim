#!/usr/bin/env python3
"""Section-0 number gate.  gpu-sim-a8x.

WHY. On 2026-10-25 four numbers in docs/training-parallelism-sim.md were found to
describe a model that no longer existed: "428 ms at 32768" for a config that had
become memory-infeasible, FSDP described as the winner at 73.05 ms after the DP
audit made it runner-up at 69.02, S_best still called path-dependent (16384 spec vs
32768 legacy) when both paths had come to agree on 32768, and a test asserting the
old threshold. Every one was caught by a human reading prose against code.

scripts/check_prose_numbers.py already enforces provenance for
experiments/*/results.md. It does NOT cover docs/, and the failure mode there is
different: the document is not quoting a stale artifact, it is quoting numbers the
CURRENT code does not produce.

WHAT THIS CHECKS. Every decimal claim inside section 0 of
docs/training-parallelism-sim.md must be one of the values computed here from
epic_cluster.parallelism_search at run time. Structural integers (sequence lengths,
rank counts, GB budgets, activation factors) are whitelisted explicitly; a decimal
that is not a computed value is a finding, whatever it means.

USAGE. python3 scripts/check_docs_model_numbers.py  (exit 1 on findings)
The computed facts are written to docs/model_facts.json so the doc has a citable
artifact for its numbers.
"""
import argparse
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from epic_cluster import link_model  # noqa: E402
from epic_cluster import parallelism_search as ps  # noqa: E402

DOC = os.path.join(ROOT, "docs/training-parallelism-sim.md")
FACTS_JSON = os.path.join(ROOT, "docs/model_facts.json")


def _diff_fact_keys(a, b):
    """Key paths where two facts-JSON texts disagree, for a readable FAIL."""
    try:
        da, db = json.loads(a), json.loads(b)
    except Exception:
        return ["<unparseable facts; run with --write>"]
    out = []

    def walk(x, y, path):
        if type(x) is not type(y):
            out.append(path or "<root>")
            return
        if isinstance(x, dict):
            for k in sorted(set(x) | set(y)):
                p = f"{path}.{k}" if path else k
                if k not in x or k not in y:
                    out.append(p)
                else:
                    walk(x[k], y[k], p)
        elif isinstance(x, list):
            if len(x) != len(y):
                out.append(f"{path} (len {len(x)} -> {len(y)})")
            else:
                for i, (xi, yi) in enumerate(zip(x, y)):
                    walk(xi, yi, f"{path}[{i}]")
        elif x != y:
            out.append(path or "<root>")

    walk(da, db, "")
    return out or ["<contents differ but no key path isolated>"]
M8 = {"n_layers": 32, "d_model": 4096, "n_heads": 32, "n_kv_heads": 8, "head_dim": 128,
      "d_ff": 14336, "vocab": 128256}
SPEC = {8: "specs/h100_rack.json", 16: "specs/h100_rack.json", 32: "specs/h100_rack_4node.json"}
ACT_FACTOR = 2.3
PEAK = 989e12


def hw_for(N, path="spec", cpg="crude", act=ACT_FACTOR, peak=None):
    peak = PEAK if peak is None else peak
    if N == 8 and path == "legacy":
        hw = {"peak_flops": peak, "mem_gb": 80.0, "fabric": "nvlink4_h100", "cp_grad_mode": cpg}
    else:
        hw = {"peak_flops": peak, "mem_gb": 80.0, "spec_path": SPEC[N], "cp_grad_mode": cpg}
    if act is not None:
        hw["act_factor"] = act
    return hw


def search(N, seq=8192, path="spec", cpg="crude", act=ACT_FACTOR, peak=None):
    m = dict(M8)
    m["seq"] = seq
    return ps.search(N, hw_for(N, path, cpg, act, peak), m=m)


def compute_facts():
    # gpu-sim-radius: how wrong the un-checkable terms can be before the answer changes.
    import subprocess, json as _json
    _rad = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "sensitivity_radius.py"), "--json"],
                          capture_output=True, text=True, check=True)
    _eb = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "error_budget.py"), "--json"],
                         capture_output=True, text=True, check=True)
    # gpu-sim-cpgqa: measured convention gap between the model's ring CP and the harness's all_gather CP.
    def _gap(seq, kv, name):
        rows = [json.loads(l) for l in open(os.path.join(ROOT, "experiments", "gpu-sim-cpgqa", name))
                if l.strip()]
        counts = {}
        for r in rows:
            if r["op"] == "allgather" and r["rank"] == 0:
                counts[r["bytes"]] = counts.get(r["bytes"], 0) + 1
        size = [b for b, n in counts.items() if n == 4][0]      # 2 layers x (K,V)
        return round(size / ((seq // 2) * kv * 4), 4)

    _cp_gap = {"seq512_kv256": _gap(512, 256, "cp2gqa.jsonl"),
               "seq256_kv256": _gap(256, 256, "cp2s256.jsonl"),
               "seq512_kv1024": _gap(512, 1024, "cp2gqa8.jsonl")}

    # gpu-sim-9oz8: which CP convention production uses changes the N=8 ANSWER, not just its margin.
    # PyTorch CP defaults to rotate_method=ALL_GATHER (torch/distributed/tensor/experimental/
    # _context_parallel/_attention.py line 77) which uses _AllGatherRotater: "Allgather the kv and return
    # only the required kv" -- the full-sequence K/V gather this harness also does. _AllToAllRotater
    # ("Use all_to_all to send the kv to the next rank") is the opt-in ring form the model prices.
    def _answer(n, seq, k):
        import importlib
        _ps = importlib.import_module("epic_cluster.parallelism_search")
        spec = "specs/h100_rack.json" if n <= 16 else "specs/h100_rack_4node.json"
        h = {"peak_flops": 989e12, "mem_gb": 80.0, "spec_path": os.path.join(ROOT, spec),
             "term_scale": {"cp": k}}
        _, feas = _ps.search(n, h, dict(_ps.MODEL, seq=seq))
        best = feas[0]
        alt = next(c for c in feas if (c.tp, c.pp, c.cp, c.dp) != (best.tp, best.pp, best.cp, best.dp))
        return {"config": [best.tp, best.pp, best.cp, best.dp, best.zero],
                "step_ms": round(best.step_sum * 1e3, 2),
                "margin_pct": round(100 * (alt.step_sum / best.step_sum - 1), 2)}

    _CELLS = [(8, 8192), (8, 32768), (16, 8192), (32, 8192), (32, 32768)]
    _conv = {}
    for _n, _seq in _CELLS:
        _conv[f"N{_n}_seq{_seq}"] = {"ring_all_to_all": _answer(_n, _seq, 1.0),
                                     "pytorch_default_all_gather": _answer(_n, _seq, 1.9922)}

    # gpu-sim: RETRACTED values, kept here as a tripwire. Section 0 is recomputed on every run, but prose
    # outside it is not, and a retraction that only reaches section 0 leaves the old claim alive elsewhere
    # (that is exactly what happened: the 109.8-vs-21.0 CP comparison survived in section 4 after being
    # retracted, and the claims register kept publishing it too). Any occurrence anywhere in the doc fails.
    RETRACTED = {
        "109.8 MB": "retracted CP byte comparison (different quantities, direction was backwards)",
        "21.0 MB of": "retracted CP byte comparison",
        "79.96": "pre-link-rate fix step time; recomputed as 88.49",
        "4091": "pre-kvcp CP crossover; now 5956",
    }

    # gpu-sim-bubble: is the PP-heavy answer an artifact of the pessimistic micro_batches default?
    def _mb_row(n, seq, mult):
        import importlib
        _ps = importlib.import_module("epic_cluster.parallelism_search")
        spec = "specs/h100_rack.json" if n <= 16 else "specs/h100_rack_4node.json"
        h = {"peak_flops": 989e12, "mem_gb": 80.0, "spec_path": os.path.join(ROOT, spec),
             "term_scale": {"cp": 1.9922}, "micro_batches_mult": mult}
        _, feas = _ps.search(n, h, dict(_ps.MODEL, seq=seq))
        best = feas[0]
        alt = next(c for c in feas if (c.tp, c.pp, c.cp, c.dp) != (best.tp, best.pp, best.cp, best.dp))
        return {"config": [best.tp, best.pp, best.cp, best.dp, best.zero],
                "step_ms": round(best.step_sum * 1e3, 2),
                "margin_pct": round(100 * (alt.step_sum / best.step_sum - 1), 2)}

    _mb = {}
    for _n in (8, 16, 32):
        _mb[f"N{_n}"] = {f"x{m}": _mb_row(_n, 8192, m) for m in (1, 2, 4, 8, 16)}

    # gpu-sim-scaleout: the answers beyond one rack, computed by the same script that publishes them.
    import subprocess, json as _json2
    _so = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "scaleout_answers.py"),
                          "--nodes", "10,100,1000", "--json"], capture_output=True, text=True, check=True)
    _scaleout = _json2.loads(_so.stdout)

    def _so_cell(fabric, model, conv):
        return _scaleout[fabric]["models"][model][conv]

    # gpu-sim-gvg5: one P for every floor. The legacy pair below used nominal 405e9
    # (70.9) while scaleout_opt uses actual GQA-aware P (405.849e9); actual wins.
    from scripts.scaleout_answers import MODELS as _facts_models
    from scripts.scaleout_answers import build_spec as _facts_spec
    from scripts.scaleout_answers import PEAK as _facts_peak, HBM_GB as _facts_hbm
    from scripts.scaleout_answers import CONVENTIONS as _facts_conv
    _p405_actual = ps.total_params(_facts_models["llama3-405b"])

    def _pp1_act():
        # gpu-sim-vvty: every published pp>1 winner re-scored through the production
        # score path at recompute=0 (the published default). Activation GB/stage vs
        # the per-GPU budget decides whether full storage is a demonstrated regime
        # (document) or an infeasible one (charge required). Cells come from the
        # already-computed _scaleout matrix -- no re-search, same winners.
        _rows = []
        for _fab, _blob in _scaleout.items():
            _spec = _facts_spec(int(_fab.split("x")[0]))
            for _mname, _cells in _blob["models"].items():
                _model = _facts_models[_mname]
                for _key, _r in _cells.items():
                    if _r is None or _r.get("pp", 0) <= 1:
                        continue
                    _batch = int(_key.split("batch=")[1]) if "batch=" in _key else 1
                    _conv = _key.split(", batch=")[0]
                    _hw = {"peak_flops": _facts_peak, "mem_gb": _facts_hbm, "spec": _spec,
                           "term_scale": {"cp": _facts_conv[_conv]},
                           "batch": _batch, "recompute": 0.0}
                    _c = ps.score(_r["tp"], _r["pp"], _r["cp"], _r["dp"], _r["zero"],
                                  _hw, _model)
                    _rows.append({"fabric": _fab, "model": _mname, "convention": _conv,
                                  "batch": _batch, "config": _r["config"],
                                  "act_gb": round(_c.mem_act_gb, 2),
                                  "headroom_gb": round(_facts_hbm - _c.mem_act_gb, 2)})
        _rows.sort(key=lambda _d: _d["headroom_gb"])
        return {"n_rows": len(_rows), "all_fit": all(_d["headroom_gb"] > 0 for _d in _rows),
                "tightest": _rows[0] if _rows else None}

    def _opt_styles():
        # gpu-sim-scup: the optimizer-style table recomputed end-to-end (dp=1
        # deltas against the default 16 B/param total) plus the 405B@80 rows per
        # style. Floors use the ACTUAL GQA-aware P (a5ro), not the nominal 405e9
        # of the fp32_master stem: the sentence quotes per-style floors.
        import importlib as _il
        _ps = _il.import_module("epic_cluster.parallelism_search")
        from scripts.scaleout_answers import MODELS as _m
        _m8 = _m["llama3-8b"]
        _p8 = _ps.total_params(_m8)
        _p405 = _ps.total_params(_m["llama3-405b"])
        _base = _ps.score(4, 1, 1, 1, 0,
                          {"peak_flops": 989e12, "mem_gb": 80.0,
                           "spec_path": os.path.join(ROOT, "specs/h100_rack.json")}, _m8).memory_gb

        def _total(state, peak):
            # dp=1, tp=4: the delta sits on the P/4 shard, so divide by that.
            got = _ps.score(4, 1, 1, 1, 0,
                             {"peak_flops": 989e12, "mem_gb": 80.0,
                              "spec_path": os.path.join(ROOT, "specs/h100_rack.json"),
                              "opt_state": state, "opt_peak": peak}, _m8).memory_gb
            return round(16 + (got - _base) / (_p8 / 4) * 1e9, 1)

        def _run(state, peak):
            _o = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "scaleout_answers.py"),
                                 "--nodes", "10", "--models", "llama3-405b",
                                 "--opt-state", state, "--opt-peak", peak, "--json"],
                                capture_output=True, text=True, check=True)
            return _json2.loads(_o.stdout)

        def _row(db, conv):
            return db["10x8"]["models"]["llama3-405b"][conv]

        _bf16 = _run("bf16_moments", "default")
        _fused = _run("fp32_moments", "fused")
        _dflt = _run("fp32_moments", "default")
        _stdy = _run("fp32_moments", "none")
        _bcell = _row(_bf16, "default all_gather")
        _f81 = round(16 * _p405 / 80 / 1e9, 1)
        out = {
            # measured rig totals, recomputed from the model (dp=1 deltas):
            # bf16 6/10, fp32 moments 12 steady, 20 default-peak, 16 fused-peak.
            "style_bytes": {
                "bf16_moments": {"steady": _total("bf16_moments", "none"),
                                   "peak_default": _total("bf16_moments", "default"),
                                   "peak_fused": _total("bf16_moments", "fused")},
                "fp32_moments": {"steady": _total("fp32_moments", "none"),
                                   "peak_default": _total("fp32_moments", "default"),
                                   "peak_fused": _total("fp32_moments", "fused")},
                "fp32_master": {"steady": _total("fp32_master", "none"),
                                  "peak_default": _total("fp32_master", "default"),
                                  "peak_fused": _total("fp32_master", "fused")},
            },
            "floor_gb_405b_at_80": {
                "bf16_moments_peak": round(10 * _p405 / 80 / 1e9, 1),
                "fp32_moments_steady": round(12 * _p405 / 80 / 1e9, 1),
                "fp32_moments_fused": round(16 * _p405 / 80 / 1e9, 1),
                "fp32_moments_default": round(20 * _p405 / 80 / 1e9, 1),
            },
            "fp32mom_fused_over_pct": round((_f81 - 80) / 80 * 100, 1),
            "bf16_80_feasible": _bcell is not None,
            "fp32mom_fused_80_feasible": _row(_fused, "default all_gather") is not None,
            "fp32mom_default_80_feasible": _row(_dflt, "default all_gather") is not None,
            "fp32mom_steady_80_feasible": _row(_stdy, "default all_gather") is not None,
        }
        if _bcell is not None:
            out["bf16_80_best_config"] = [_bcell["tp"], _bcell["pp"], _bcell["cp"],
                                             _bcell["dp"], _bcell["zero"]]
            out["bf16_80_best_step_ms"] = round(1000 * _bcell["step_s"])
            out["bf16_80_best_mem_gb"] = _bcell["mem_gb"]
            out["bf16_80_margin"] = _bcell["margin_pct"]
        return out

    facts = {"generated_by": "scripts/check_docs_model_numbers.py",
             "scaleout": {
                 "n70b_80gpus_ring_margin": _so_cell("10x8", "llama3-70b", "ring all_to_all")["margin_pct"],
                 "n70b_80gpus_ag_margin": _so_cell("10x8", "llama3-70b", "default all_gather")["margin_pct"],
                 "n70b_80gpus_ag_step_ms": round(1000 * _so_cell("10x8", "llama3-70b", "default all_gather")["step_s"], 1),
                 "n8b_80gpus_ag_margin": _so_cell("10x8", "llama3-8b", "default all_gather")["margin_pct"],
                 "worst_margin_8000": min(
                     (_so_cell("1000x8", m, c)["margin_pct"] or 0.0)
                     for m in ("llama3-8b", "llama3-70b", "llama3-405b")
                     for c in ("ring all_to_all", "default all_gather")),
                 "best_margin_8000": max(
                     (_so_cell("1000x8", m, c)["margin_pct"] or 0.0)
                     for m in ("llama3-8b", "llama3-70b", "llama3-405b")
                     for c in ("ring all_to_all", "default all_gather")),
                 "n405b_80gpus_feasible": _so_cell("10x8", "llama3-405b", "default all_gather") is not None,
                 # why 405B does not fit at 80 GPUs: fp32 master + fp32 Adam m/v + bf16 grads = 14 B/param,
                 # and P/N is fixed by the GPU count no matter how the 80 GPUs are decomposed.
                 # P is the actual GQA-aware count (gpu-sim-gvg5), like the scaleout_opt floors.
                 "opt_state_floor_gb_405b_at_80": round(14 * _p405_actual / 80 / 1e9, 1),
                 "opt_state_floor_gb_405b_at_800": round(14 * _p405_actual / 800 / 1e9, 1),
                 "n405b_800gpus_ag_margin": _so_cell("100x8", "llama3-405b", "default all_gather")["margin_pct"],
             },
             "scaleout_opt": _opt_styles(),
             "pp1_act": _pp1_act(),
             "micro_batches_sensitivity": _mb,
             "retracted_values": RETRACTED,
             "cp_convention_answers": _conv,
             "cp_convention_gap": _cp_gap,
             "sensitivity_radius": _json.loads(_rad.stdout),
             "error_budget": _json.loads(_eb.stdout),
             "model": "llama3-8b", "act_factor": ACT_FACTOR,
             "pricing_path": "spec/topology", "rank_by": "sum", "rows": {}}

    # gpu-sim-hz3u: the 405B matrix with FULL activation recompute, under the measured knob
    # (float 1.0: keep 1-0.67 of activations, 1.25x compute -- experiments/gpu-sim-recompute/).
    # Default stays 0.0, so every number above is untouched; these facts record the
    # sensitivity, not a revision.
    _sof = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "scaleout_answers.py"),
                           "--nodes", "10,100,1000", "--models", "llama3-405b",
                           "--recompute", "1.0", "--json"],
                          capture_output=True, text=True, check=True)
    _full = _json2.loads(_sof.stdout)

    def _sof_cell(fabric, conv):
        return _full[fabric]["models"]["llama3-405b"][conv]

    def _row(cell):
        if cell is None:
            return {"feasible": False}
        return {"feasible": True, "config": cell["config"],
                "step_s": cell["step_s"], "margin_pct": cell["margin_pct"],
                "mem_gb": cell["mem_gb"]}

    facts["recompute_sensitivity_405b"] = {
        "setting": ("float 0.0..1.0, default 0.0; 1.0 keeps (1-0.67) of activations + "
                    "1.25x compute (measured, experiments/gpu-sim-recompute/)"),
        "feasible_80_full": _sof_cell("10x8", "default all_gather") is not None,
        "winner_800_full": {c: _row(_sof_cell("100x8", c))
                              for c in ("ring all_to_all", "default all_gather")},
        "winner_8000_full": {c: _row(_sof_cell("1000x8", c))
                               for c in ("ring all_to_all", "default all_gather")},
    }

    # gpu-sim-hz3u acceptance cell (one GPU-count, one batch, both settings, same model on
    # both sides so the comparison carries the same time charge): 70B at 80 GPUs, batch=32.
    # With no recompute this must reproduce the old pp=4 row; with full recompute the model
    # must stop buying pipeline stages and buy data parallelism instead (pure FSDP).
    from scripts.scaleout_answers import build_spec as _build_spec
    from scripts.scaleout_answers import MODELS as _SO_MODELS
    _spec10 = _build_spec(10)
    _r70 = {}
    for _rec in (0.0, 1.0):
        _res, _feas = ps.search(80, {"peak_flops": PEAK, "mem_gb": 80.0, "spec": _spec10,
                                     "term_scale": {"cp": 1.9922}, "batch": 32,
                                     "recompute": _rec}, _SO_MODELS["llama3-70b"])
        _feas = [c for c in _feas if c.tp <= 8 and c.pp <= _SO_MODELS["llama3-70b"]["n_layers"]]
        _b = _feas[0]
        _a = next(c for c in _feas
                  if (c.tp, c.pp, c.cp, c.dp) != (_b.tp, _b.pp, _b.cp, _b.dp))
        _r70[str(_rec)] = {"config": f"tp={_b.tp} pp={_b.pp} cp={_b.cp} dp={_b.dp} zero={_b.zero}",
                           "step_s": round(_b.step_sum, 3),
                           "margin_pct": round(100 * (_a.step_sum / _b.step_sum - 1), 2),
                           "mem_gb": round(_b.memory_gb, 1)}
    facts["recompute_70b_b32_80"] = _r70

    # gpu-sim-wv2 (partial): the compute term divides FLOPs by peak with no efficiency
    # factor. Record the implied MFU and the step at a measured (0.566) and pessimistic
    # (0.40) efficiency, so the doc can quote them and the numbers are reproducible here.
    facts["mfu_sensitivity"] = {}
    _implied = None
    for tag, mfu in (("implied_0.786", None), ("measured_5090", 0.566), ("typical_h100", 0.40)):
        peak = PEAK if mfu is None else PEAK * mfu / 0.786
        _, feas = search(8, peak=peak)
        b = feas[0]
        facts["mfu_sensitivity"][tag] = {"mfu": round(mfu, 3) if mfu else None,
                                         "config": [b.tp, b.pp, b.cp, b.dp, b.zero],
                                         "step_ms": round(b.step_sum * 1e3, 2)}
        if mfu is None:
            _implied = (6 * 8e9 * 8192) / (8 * PEAK) / b.compute
            facts["mfu_sensitivity"][tag]["implied_mfu"] = round(_implied, 3)
            facts["mfu_sensitivity"]["implied_mfu_pct"] = round(_implied * 100, 1)
            _base = b.step_sum
    for tag, key in (("measured_5090", "ratio_at_measured"), ("typical_h100", "ratio_at_typical")):
        facts["mfu_sensitivity"][key] = round(
            facts["mfu_sensitivity"][tag]["step_ms"] / facts["mfu_sensitivity"]["implied_0.786"]["step_ms"], 2)

    # gpu-sim-mfum: how DECISIVE is each winner? The margin against the best config with a
    # different (tp,pp,cp,dp) shape, and where the N=16 answer stops holding. A tie between
    # zero policies (same shape, zero=2 vs zero=3) is not a fragile win, so the runner-up is
    # taken over shapes, not over rows.
    def _winner(n, mfu):
        _, feas = search(n, peak=PEAK * mfu / 0.786)
        best = feas[0]
        shape = (best.tp, best.pp, best.cp, best.dp)
        alt = next(c for c in feas if (c.tp, c.pp, c.cp, c.dp) != shape)
        return (best.tp, best.pp, best.cp, best.dp, best.zero), best.step_sum, \
               (alt.tp, alt.pp, alt.cp, alt.dp, alt.zero), alt.step_sum

    # gpu-sim-hroom: does the answer depend on treating HBM as a HARD fit? Sweep a required
    # headroom (fraction of HBM kept free) over the published grid and record where it bites.
    # Exactly one cell changes: N=16 seq 32768, whose winner otherwise occupies 94% of HBM.
    facts["headroom_sensitivity"] = {"note": "winner at required free-HBM fraction 0 / 0.10 / 0.20, "
                                             "evaluated at the MEASURED 0.566 efficiency (the "
                                             "implied 0.786 winner already has headroom)",
                                     "grid": {}}
    for n in (8, 16, 32):
        for seq in (8192, 16384, 32768):
            cell = {}
            for hr in (0.0, 0.10, 0.20):
                m = dict(M8)
                m["seq"] = seq
                hwv = {"peak_flops": PEAK * 0.566 / 0.786, "mem_gb": 80.0,
                       "spec_path": SPEC[n], "mem_headroom": hr}
                _, feas = ps.search(n, hwv, m)
                cell[f"hr{hr:.2f}"] = ({"config": [feas[0].tp, feas[0].pp, feas[0].cp,
                                                   feas[0].dp, feas[0].zero],
                                        "step_ms": round(feas[0].step_sum * 1e3, 2),
                                        "mem_gb": round(feas[0].memory_gb, 1)}
                                       if feas else None)
            shapes = {tuple(v["config"][:4]) for v in cell.values() if v}
            cell["changes"] = len(shapes) > 1
            facts["headroom_sensitivity"]["grid"][f"N{n}_seq{seq}"] = cell

    # gpu-sim-scross: where does CONTEXT PARALLELISM start paying? The docs say "CP pays at
    # seq 8192" and S_best=8192, but 8192 was the smallest grid point, not a computed boundary.
    # Bisect the seq at which the best cp>1 config overtakes the best cp=1 config: the crossover
    # sits WELL BELOW 8192, so S_best is a discretization with room to spare, not a knife edge.
    def _cp_advantage(seq, mfu):
        m = dict(M8)
        m["seq"] = seq
        hwv = {"peak_flops": PEAK * mfu / 0.786, "mem_gb": 80.0,
               "spec_path": "specs/h100_rack.json"}
        _, feas = ps.search(8, hwv, m)
        cp1 = [c for c in feas if c.cp == 1]
        cpg = [c for c in feas if c.cp > 1]
        if not cp1 or not cpg:
            return None
        return 100 * (min(c.step_sum for c in cp1) / min(c.step_sum for c in cpg) - 1)

    facts["cp_crossover"] = {}
    for tag, mfu in (("implied", 0.786), ("measured", 0.566), ("typical", 0.40)):
        lo, hi = 2048, 8192
        for _ in range(22):
            mid = (lo + hi) / 2
            a = _cp_advantage(int(mid), mfu)
            if a is None or a > 0:
                hi = mid
            else:
                lo = mid
        facts["cp_crossover"][f"{tag}_seq"] = int((lo + hi) / 2)
        facts["cp_crossover"][f"advantage_at_8192_{tag}_pct"] = round(
            _cp_advantage(8192, mfu), 2)
        facts["cp_crossover"][f"advantage_at_4096_{tag}_pct"] = round(
            _cp_advantage(4096, mfu), 2)

    # gpu-sim-frag: WHERE is the recommendation a coin flip? Margin of the winner against the
    # best config with a different (tp,pp,cp,dp) shape, over N x seq x MFU. This is the grid
    # that showed the fragility is a LONG-SEQUENCE phenomenon, not an N=16 quirk: at 8192 only
    # N=16 at a pessimistic MFU is tight, but at 16384/32768 the N=8 answer itself wins by ~1%.
    facts["fragility_map"] = {"note": "margin = best config with a different shape; <2% = coin flip",
                              "grid": {}}
    for n in (8, 16, 32):
        for seq in (8192, 16384, 32768):
            cell = {}
            for tag, mfu in (("implied", 0.786), ("measured", 0.566), ("typical", 0.40)):
                m = dict(M8)
                m["seq"] = seq
                hwv = {"peak_flops": PEAK * mfu / 0.786, "mem_gb": 80.0, "spec_path": SPEC[n]}
                _, feas = ps.search(n, hwv, m)
                if not feas:
                    cell[tag] = None
                    continue
                b = feas[0]
                alt = next((c for c in feas
                            if (c.tp, c.pp, c.cp, c.dp) != (b.tp, b.pp, b.cp, b.dp)), None)
                cell[tag] = {"config": [b.tp, b.pp, b.cp, b.dp],
                             "step_ms": round(b.step_sum * 1e3, 2),
                             "margin_pct": round(100 * (alt.step_sum / b.step_sum - 1), 2)
                             if alt else None}
            facts["fragility_map"]["grid"][f"N{n}_seq{seq}"] = cell

    # gpu-sim-aknob: act_factor is a MEMORY knob -- record where, if anywhere, it moves the
    # answer. A 5x band around the fit leaves the ranking untouched; the bend points and the
    # all-infeasible ceilings are computed here so the doc cannot quote drift.
    facts["act_factor_knob"] = {"calibrated": 2.3,
                                "band_tested": [1.0, 12.0],
                                "bend_point": {}, "infeasible_above": {}}
    for n in (8, 16, 32):
        _, base = search(n)
        base_shape = (base[0].tp, base[0].pp, base[0].cp, base[0].dp, base[0].zero)

        def _w(a):
            spec = SPEC[n]
            hwv = {"peak_flops": PEAK, "mem_gb": 80.0, "spec_path": spec, "act_factor": a}
            _, feas = ps.search(n, hwv, m=ps.MODEL)
            return feas

        a, bend = 2.3, None
        while a < 200:
            feas = _w(a)
            if not feas:
                break
            b = feas[0]
            if (b.tp, b.pp, b.cp, b.dp, b.zero) != base_shape:
                bend = round(a, 1)
                break
            a *= 1.15
        facts["act_factor_knob"]["bend_point"][f"N{n}"] = bend
        lo, hi = 1.0, 400.0
        for _ in range(28):
            mid = (lo + hi) / 2
            if _w(mid):
                lo = mid
            else:
                hi = mid
        facts["act_factor_knob"]["infeasible_above"][f"N{n}"] = round(lo, 1)

    facts["mfu_margin_sweep"] = {"note": "margin = best config with a different shape"}
    for n in (8, 16, 32):
        row = {}
        for tag, mfu in (("implied", 0.786), ("measured", 0.566), ("typical_h100", 0.40)):
            w, ms, alt, alt_ms = _winner(n, mfu)
            row[tag] = {"mfu": mfu, "config": list(w), "step_ms": round(ms * 1e3, 2),
                        "runner_shape": list(alt), "margin_pct": round(100 * (alt_ms / ms - 1), 2)}
        facts["mfu_margin_sweep"][f"N{n}"] = row

    # where does the N=16 "no tensor parallelism" answer stop holding? Bisect on MFU.
    lo, hi = 0.25, 0.40
    for _ in range(22):
        mid = (lo + hi) / 2
        if _winner(16, mid)[0][0] == 1:          # still tp=1 -> the answer holds
            hi = mid
        else:
            lo = mid
    facts["mfu_margin_sweep"]["N16_crossover_mfu"] = round(hi, 3)

    # 1) winners + top-3 at seq 8192, N=8/16/32
    for N in (8, 16, 32):
        _, feas = search(N)
        top = feas[:3]
        facts["rows"][f"N{N}"] = {
            "winner": {"config": [top[0].tp, top[0].pp, top[0].cp, top[0].dp, top[0].zero],
                       "step_ms": round(top[0].step_sum * 1e3, 2)},
            "top3": [{"config": [c.tp, c.pp, c.cp, c.dp, c.zero],
                      "step_ms": round(c.step_sum * 1e3, 2)} for c in top],
        }
        fsdp = [c for c in feas if (c.tp, c.pp, c.cp, c.dp, c.zero) == (1, 1, 1, 8, 3)]
        facts["rows"][f"N{N}"]["winner"]["mem_gb"] = round(top[0].memory_gb, 1)
        if len(feas) and N == 8:
            facts["rows"]["N8"]["fsdp_zero3_row"] = {
                "step_ms": round(fsdp[0].step_sum * 1e3, 2),
                "mem_gb": round(fsdp[0].memory_gb, 1)} if fsdp else None

    # 1b) the FSDP row's memory at seq 16384 (the claim "85.0 GB > 80")
    m = dict(M8)
    m["seq"] = 16384
    fsdp16k = ps.score(1, 1, 1, 8, 3, hw_for(8), m)
    facts["fsdp_mem_16384_gb"] = round(fsdp16k.memory_gb, 1)

    # 1c) the legacy single-fabric path at 8192/16384/32768 (section 0 quotes it beside
    # the spec numbers), and the dp=8 rows' step/memory/dp_comm that the answer cites.
    facts["legacy_path_steps"] = {}
    for S in (8192, 16384, 32768):
        m = dict(M8)
        m["seq"] = S
        hw = {"peak_flops": 989e12, "mem_gb": 80.0, "fabric": "nvlink4_h100",
              "cp_grad_mode": "crude", "act_factor": ACT_FACTOR}
        _, f = ps.search(8, hw, m)
        facts["legacy_path_steps"][str(S)] = round(f[0].step_sum * 1e3, 2)
    facts["spec_path_steps"] = {}
    for S in (16384, 32768):
        m = dict(M8)
        m["seq"] = S
        _, f = ps.search(8, hw_for(8), m)
        facts["spec_path_steps"][str(S)] = round(f[0].step_sum * 1e3, 2)
    dp8 = {}
    for z in (2, 3):
        m8 = dict(M8)
        m8["seq"] = 8192
        c = ps.score(1, 1, 1, 8, z, hw_for(8), m8)
        dp8[f"zero={z}"] = {"step_ms": round(c.step_sum * 1e3, 2),
                            "mem_gb": round(c.memory_gb, 1),
                            "dp_comm_ms": round(c.dp_comm * 1e3, 2)}
    facts["dp8_rows"] = dp8

    # 1c2) the rows the CURRENT ANSWER paragraph quotes beside the winner: the runner-up
    # (best non-winning config) and the best config with dp>1 at N=8, both under hw_for(8).
    _, _f8 = ps.search(8, hw_for(8), dict(M8, seq=8192))
    _ru, _dp1 = _f8[1], next(c for c in _f8 if c.dp > 1)
    facts["n8_runnerup"] = {"config": [_ru.tp, _ru.pp, _ru.cp, _ru.dp, _ru.zero],
                              "step_ms": round(_ru.step_sum * 1e3, 2),
                              "mem_gb": round(_ru.memory_gb, 1)}
    facts["n8_dp1_best"] = {"config": [_dp1.tp, _dp1.pp, _dp1.cp, _dp1.dp, _dp1.zero],
                               "step_ms": round(_dp1.step_sum * 1e3, 2),
                               "mem_gb": round(_dp1.memory_gb, 1)}

    # 1d) TP payload (gpu-sim-6wd): four_d is MEASURED on the harness (B*seq*d per
    # all-reduce, identical at tp=2 and tp=4) and derived for Megatron (d_ff never
    # crosses a rank boundary), so it is the default and the refuted alternatives must
    # stay strictly dearer/cheaper than it.
    facts["tp_payload_winners"] = {}
    for N in (8, 16, 32):
        hw = {"peak_flops": 989e12, "mem_gb": 80.0, "spec_path": SPEC[N],
              "cp_grad_mode": "crude", "act_factor": ACT_FACTOR}
        mm = dict(M8)
        mm["seq"] = 8192
        _, ff = ps.search(N, hw, mm)
        b = ff[0]
        facts["tp_payload_winners"][f"N{N}"] = {"config": [b.tp, b.pp, b.cp, b.dp, b.zero],
                                                "step_ms": round(b.step_sum * 1e3, 2)}

    # 1g) alpha piece-charging (gpu-sim-8nt follow-up): pay alpha P times on payloads
    # >= 256 KiB and the winner must not move. Reproduces
    # experiments/gpu-sim-8nt/alpha_piece_charging.py without importing it.
    facts["alpha_piece_charging"] = {}
    _orig_predict = link_model.predict
    for pieces in (1, 4, 8):
        def patched(spec, algorithm, count, dtype, _p=pieces, **kw):
            kw["pieces_per_phase"] = 1
            out = _orig_predict(spec, algorithm, count, dtype, **kw)
            nb = count * link_model.DTYPE_BYTES.get(dtype, 2)
            if _p > 1 and nb >= 256 * 1024:
                out["duration_s"] += (_p - 1) * (out["_debug"]["lat_term"] or 0.0)
            return out
        link_model.predict = patched
        hw = {"peak_flops": 989e12, "mem_gb": 80.0, "spec_path": SPEC[8],
              "cp_grad_mode": "crude", "act_factor": ACT_FACTOR}
        mm = dict(M8)
        mm["seq"] = 8192
        try:
            _, ff = ps.search(8, hw, mm)
            b = ff[0]
            facts["alpha_piece_charging"][str(pieces)] = {
                "config": [b.tp, b.pp, b.cp, b.dp, b.zero],
                "step_ms": round(b.step_sum * 1e3, 2),
                "runner_up_ms": round(ff[1].step_sum * 1e3, 2) if len(ff) > 1 else None}
        finally:
            link_model.predict = _orig_predict

    # 2) seq 32768: the dp=8 zero=2 config must be infeasible, and name the best feasible
    _, feas32k = search(8, seq=32768)
    best = feas32k[0]
    facts["seq32768"] = {
        "best_config": [best.tp, best.pp, best.cp, best.dp, best.zero],
        "best_step_ms": round(best.step_sum * 1e3, 2),
        "dp8_zero2_feasible": any(
            (c.tp, c.pp, c.cp, c.dp, c.zero) == (1, 1, 1, 8, 2) for c in feas32k),
    }

    # 3) S_best matrix at N=8 (the claim that the path no longer matters)
    seqs = [8192, 12288, 16384, 20480, 24576, 32768, 65536]
    sb = {}
    for path in ("legacy", "spec"):
        for cpg in ("crude", "production"):
            for act in (2.3, 1.0):
                s_best = None
                for seq in seqs:
                    try:
                        _, f = search(8, seq=seq, path=path, cpg=cpg, act=act)
                    except ValueError:
                        f = []
                    if f and f[0].cp > 1:
                        s_best = seq
                        break
                sb[f"path={path},cp_grad={cpg},act={act},N=8"] = s_best
    facts["s_best"] = sb

    # 3b) compute floors (the attention-FLOPs row quotes them) and the 70B/32 row at 32768
    for S in (8192, 32768):
        m = dict(M8)
        m["seq"] = S
        c = ps.score(1, 1, 1, 8, 2, hw_for(8), m)
        facts[f"compute_floor_{S}_ms"] = round(c.compute * 1e3, 2)
    m70 = {"n_layers": 80, "d_model": 8192, "n_heads": 64, "n_kv_heads": 8, "head_dim": 128,
           "d_ff": 28672, "vocab": 128256, "seq": 32768}
    hw32 = {"peak_flops": 989e12, "mem_gb": 80.0, "spec_path": "specs/h100_rack_4node.json",
            "cp_grad_mode": "crude", "act_factor": ACT_FACTOR}
    r70, f70 = ps.search(32, hw32, m70)
    facts["llama3-70b_N32_seq32768_nconfigs"] = len(r70)
    if f70:
        facts["llama3-70b_N32_seq32768_step_s"] = round(f70[0].step_sum, 2)
        facts["llama3-70b_N32_seq32768_mem_gb"] = round(f70[0].memory_gb, 1)
        facts["llama3-70b_N32_seq32768_nfeasible"] = len(f70)
        facts["llama3-70b_N32_seq32768_config"] = [f70[0].tp, f70[0].pp, f70[0].cp, f70[0].dp, f70[0].zero]

    # 4) model-size sweep (the "does the answer generalize" table)
    sweep_models = {
        "llama3-8b": M8,
        "qwen2.5-32b": {"n_layers": 64, "d_model": 5120, "n_heads": 40, "n_kv_heads": 8,
                        "head_dim": 128,
                        "d_ff": 27648, "vocab": 152064},
        "llama2-34b": {"n_layers": 48, "d_model": 8192, "n_heads": 64, "n_kv_heads": 64,
                       "head_dim": 128,
                       "d_ff": 22016, "vocab": 32000},
        "llama3-70b": {"n_layers": 80, "d_model": 8192, "n_heads": 64, "n_kv_heads": 8,
                       "head_dim": 128,
                       "d_ff": 28672, "vocab": 128256},
    }
    facts["sweep"] = {}
    for name, mdl in sweep_models.items():
        row = {}
        for N in (8, 16):
            m = dict(mdl)
            m["seq"] = 8192
            hw = {"peak_flops": 989e12, "mem_gb": 80.0,
                  "spec_path": "specs/h100_rack_4node.json" if N == 32 else "specs/h100_rack.json",
                  "cp_grad_mode": "crude", "act_factor": ACT_FACTOR}
            _, feas = ps.search(N, hw, m)
            # the binding config when the model does not fit: minimum memory in the space
            allcfg = []
            for tp in (1, 2, 4, 8):
                for pp in (1, 2, 4, 8):
                    for cp in (1, 2, 4, 8):
                        for dp in (1, 2, 4, 8, 16, 32):
                            if tp * pp * cp * dp != N:
                                continue
                            for zero in (0, 1, 2, 3):
                                try:
                                    allcfg.append(ps.score(tp, pp, cp, dp, zero, hw, m))
                                except ValueError:
                                    pass
            argmin = min(allcfg, key=lambda c: c.memory_gb)
            entry = {
                "n_ranks": N,
                "min_mem_gb": round(argmin.memory_gb, 2),
                "margin_gb": round(argmin.memory_gb - 80.0, 2),
                "min_mem_config": [argmin.tp, argmin.pp, argmin.cp, argmin.dp, argmin.zero],
                "breakdown_gb": {
                    "params": round(argmin.mem_params_gb, 2),
                    "grads": round(argmin.mem_grads_gb, 2),
                    "master": round(argmin.mem_master_gb, 2),
                    "opt": round(argmin.mem_opt_gb, 2),
                    "act": round(argmin.mem_act_gb, 2),
                    "logits": round(argmin.mem_logits_gb, 2),
                },
            }
            if feas:
                b = feas[0]
                entry["best_config"] = [b.tp, b.pp, b.cp, b.dp, b.zero]
                entry["best_step_ms"] = round(b.step_sum * 1e3, 2)
                entry["best_mem_gb"] = round(b.memory_gb, 2)
            row[f"N{N}"] = entry
        facts["sweep"][name] = row
    return facts


def section0(text):
    """The decisive parts of section 0.

    Two disjoint segments, on purpose: the CURRENT ANSWER / ranking / S_best text and
    the model-size table. The MFU-overlap analysis between them carries numbers from
    other beads' artifacts (weight-only times, calibration factors) that this gate
    cannot re-derive; those are covered by scripts/check_prose_numbers.py on their own
    results documents.
    """
    table_start = text.index("| cell |")
    table_end = text.index("\n\n", table_start)
    table_block = text[table_start:table_end]
    s1 = text[text.index("**CURRENT ANSWER"):text.index("THE RANKING IS A NO-OVERLAP RANKING")]
    s2 = text[text.index("**Does the answer generalize past 8B?"):text.index("## 1. Premise")]
    return table_block + "\n" + s1 + "\n" + s2


# Structural integers the doc legitimately uses; not model outputs.
STRUCTURAL = {
    "8192", "12288", "16384", "20480", "24576", "32768", "65536", "131072",
    "8", "16", "32", "80", "2.3", "1.0", "0.0", "3.0",
    "1", "2", "3", "4", "6", "12", "24", "27", "100", "107", "30.6",
}


def allowed_decimals(facts):
    vals = set()

    def add(v):
        if isinstance(v, (int, float)):
            s = f"{v:.2f}" if isinstance(v, float) else str(v)
            vals.add(s)
            vals.add(f"{v:.1f}")
            if isinstance(v, float):
                vals.add(f"{v:.3f}")   # efficiencies are quoted to three places

    for row in facts["rows"].values():
        if not row:
            continue
        for k in ("winner", "fsdp_zero3_row"):
            if row.get(k):
                add(row[k]["step_ms"])
                if "mem_gb" in row[k]:
                    add(row[k]["mem_gb"])
        for c in row.get("top3", []):
            add(c["step_ms"])
    add(facts["seq32768"]["best_step_ms"])
    add(facts["fsdp_mem_16384_gb"])
    for S in (8192, 32768):
        add(facts[f"compute_floor_{S}_ms"])
    if "llama3-70b_N32_seq32768_step_s" in facts:
        add(facts["llama3-70b_N32_seq32768_step_s"])
        add(facts["llama3-70b_N32_seq32768_mem_gb"])
    for v in facts.get("legacy_path_steps", {}).values():
        add(v)
    for v in facts.get("spec_path_steps", {}).values():
        add(v)
    _apc = facts["alpha_piece_charging"]
    for v in _apc.values():
        add(v["step_ms"])
        if v["runner_up_ms"] is not None:
            add(v["runner_up_ms"])
    # the relative slowdown quoted in prose (+6.7% at P=8)
    add(round(100 * (_apc["8"]["step_ms"] / _apc["1"]["step_ms"] - 1), 1))
    add(round(100 * (_apc["4"]["step_ms"] / _apc["1"]["step_ms"] - 1), 1))
    for v in facts["tp_payload_winners"].values():
        add(v["step_ms"])
    for k, v in facts.get("mfu_sensitivity", {}).items():
        if isinstance(v, dict):
            add(v["step_ms"])
            for f in ("mfu", "implied_mfu"):
                if v.get(f) is not None:
                    add(v[f])
                    add(round(v[f] * 100, 1))    # efficiencies are also quoted as percentages
        else:
            add(v)
    _ms = facts.get("mfu_margin_sweep", {})
    for _k, _v in _ms.items():
        if isinstance(_v, dict):
            for _tag in ("implied", "measured", "typical_h100"):
                if _tag in _v:
                    add(_v[_tag]["step_ms"])
                    add(_v[_tag]["margin_pct"])
    for _k, _v in facts.get("cp_crossover", {}).items():
        add(_v)
        if isinstance(_v, (int, float)) and _v < 0:
            add(abs(_v))   # prose quotes a LOSS as a magnitude ("loses by 3.65%")
    _hr = facts.get("headroom_sensitivity", {}).get("grid", {})
    for _cell in _hr.values():
        for _k in ("hr0.00", "hr0.10", "hr0.20"):
            if _cell.get(_k):
                add(_cell[_k]["step_ms"])
                add(_cell[_k]["mem_gb"])
    _fm = facts.get("fragility_map", {}).get("grid", {})
    for _cell in _fm.values():
        for _tag in ("implied", "measured", "typical"):
            if _cell.get(_tag):
                add(_cell[_tag]["step_ms"])
                add(_cell[_tag]["margin_pct"])
    _ak = facts.get("act_factor_knob", {})
    add(_ak.get("calibrated"))
    for d in (_ak.get("bend_point", {}), _ak.get("infeasible_above", {})):
        for v in d.values():
            if v is not None:
                add(v)
    for v in _ak.get("band_tested", []):
        add(v)
    if "N16_crossover_mfu" in _ms:
        add(_ms["N16_crossover_mfu"])
    for v in facts.get("dp8_rows", {}).values():
        for k in ("step_ms", "mem_gb", "dp_comm_ms"):
            add(v[k])
    for v in (facts.get("n8_runnerup"), facts.get("n8_dp1_best")):
        if v:
            add(v["step_ms"])
            add(v["mem_gb"])
    for row in facts.get("sweep", {}).values():
        for k, v in row.items():
            if isinstance(v, dict):
                for kk in ("best_step_ms", "best_mem_gb", "min_mem_gb", "margin_gb"):
                    if kk in v:
                        add(v[kk])
                if "breakdown_gb" in v:
                    for bv in v["breakdown_gb"].values():
                        add(bv)
            elif isinstance(v, (int, float)):
                add(v)
    for v in facts["s_best"].values():
        if v is not None:
            add(v)
    for _k, _v in facts["scaleout"].items():
        if isinstance(_v, (int, float)) and not isinstance(_v, bool):
            add(_v)
    # gpu-sim-vvty: the tightest pp>1 activation figure the prose quotes.
    _tight = facts.get("pp1_act", {}).get("tightest", {}) or {}
    for _v in (_tight.get("act_gb"), _tight.get("headroom_gb")):
        if isinstance(_v, (int, float)):
            add(_v)
    # gpu-sim-scup: every style byte count, floor, and best-row number the paragraph quotes.
    for _v in facts.get("scaleout_opt", {}).get("style_bytes", {}).values():
        for _b in _v.values():
            add(_b)
    for _k, _v in facts.get("scaleout_opt", {}).items():
        if _k in ("style_bytes", "bf16_80_best_config"):
            continue
        if isinstance(_v, (int, float)) and not isinstance(_v, bool):
            add(_v)
    _msb = facts.get("scaleout_opt", {}).get("measured_style_bytes", {})
    for _v in _msb.get("bf16_moments", {}).values():
        if isinstance(_v, (int, float)) and not isinstance(_v, bool):
            add(_v)
    for _v in _msb.get("fp32_moments", {}).values():
        if isinstance(_v, (int, float)) and not isinstance(_v, bool):
            add(_v)
    for _v in _msb.get("measured_floor_gb_405b_at_80", {}).values():
        if isinstance(_v, (int, float)) and not isinstance(_v, bool):
            add(_v)
    # gpu-sim-hz3u: every with-recompute number the paragraph quotes -- 405B winners at
    # 800/8000 (ms with one decimal, margins/mems as stored) and the 70B/batch-32
    # acceptance cell at 80 GPUs, both settings (steps quoted in seconds as stored,
    # plus ms, margins, mems).
    _rs = facts.get("recompute_sensitivity_405b", {})
    for _w in (_rs.get("winner_800_full", {}), _rs.get("winner_8000_full", {})):
        for _row in _w.values():
            if _row.get("feasible"):
                add(round(1000 * _row["step_s"], 1))
                add(_row["step_s"])
                add(_row["margin_pct"])
                add(_row["mem_gb"])
    for _row in facts.get("recompute_70b_b32_80", {}).values():
        add(round(1000 * _row["step_s"], 1))
        add(_row["step_s"])
        add(_row["margin_pct"])
        add(_row["mem_gb"])
    for _n_rows in facts["micro_batches_sensitivity"].values():
        for _row in _n_rows.values():
            add(_row["step_ms"])
            add(_row["margin_pct"])
    for _cell in facts["cp_convention_answers"].values():
        for _row in _cell.values():
            add(_row["step_ms"])
            add(_row["margin_pct"])
    for v in facts["cp_convention_gap"].values():
        add(v)
        add(round(v, 2))
    for row in facts["sensitivity_radius"].values():
        for key in ("flip_cp_up", "flip_cp_down", "flip_pp_up", "flip_pp_down"):
            if row.get(key) is not None:
                add(row[key])
                add(round(row[key], 2))
                add(round(row[key], 1))
    # gpu-sim-i5ze: allow error-budget flip values in prose.
    _eb = facts.get("error_budget", {})
    for _r in _eb.get("ranked", []) + _eb.get("unranked", []):
        for _k in ("flip_value", "flip_up", "flip_down", "current_margin"):
            _v = _r.get(_k)
            if isinstance(_v, (int, float)) and not isinstance(_v, bool):
                add(_v)
    return vals


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="write the facts to the tracked path (main-only regeneration)")
    ap.add_argument("--full", action="store_true",
                    help="recompute all facts and compare to the committed artifact (slow)")
    args = ap.parse_args()
    full = args.full or args.write or os.environ.get("CI_GATES_FULL") == "1"

    grand = os.path.join(ROOT, "scripts/docs_model_numbers_grandfathered.txt")
    grandfathered = set()
    if os.path.exists(grand):
        for ln in open(grand):
            ln = ln.split("#")[0].strip()
            if ln:
                grandfathered.add(ln.split()[0])

    if full:
        facts = compute_facts()
        jsontext = json.dumps(facts, indent=2) + "\n"
        if args.write:
            _tmp = FACTS_JSON + ".tmp"
            with open(_tmp, "w") as fh:
                fh.write(jsontext)
            os.replace(_tmp, FACTS_JSON)
            print(f"wrote {FACTS_JSON}")
        else:
            tracked_text = open(FACTS_JSON).read() if os.path.exists(FACTS_JSON) else None
            if tracked_text is None:
                print(f"FAIL: {FACTS_JSON} absent; the committed facts are the artifact "
                      f"every prose check reads. Regenerate with --write and commit.")
                return 1
            elif tracked_text != jsontext:
                # This used to be a NOTE and exit 0, so the facts could be stale
                # against the code that produces them while every gate stayed
                # green -- exactly what happened after i5ze added budget cells and
                # model_facts was never regenerated (found in review 2026-10-28).
                # A drift that prints and passes is a comment, not a check: the
                # expensive half of the split would be unobservable on the runs
                # that pay for it.
                drifted = _diff_fact_keys(tracked_text, jsontext)
                print(f"FAIL: {FACTS_JSON} differs from the facts this run produces "
                      f"in {len(drifted)} key path(s):")
                for path in drifted[:20]:
                    print(f"  {path}")
                if len(drifted) > 20:
                    print(f"  ... and {len(drifted) - 20} more")
                print("Regenerate on main with --write and COMMIT the result; "
                      "a run that regenerates is not a verified run.")
                return 1
            else:
                print(f"OK: {FACTS_JSON} matches the facts this run produces.")
    else:
        # FAST: prose vs committed artifact only (cheap). No recompute.
        if not os.path.exists(FACTS_JSON):
            print(f"FAIL: {FACTS_JSON} missing; cannot check prose without committed artifact")
            return 1
        facts = json.load(open(FACTS_JSON))
        print("FAST MODE: checking prose against committed docs/model_facts.json "
              "(artifact-vs-code recompute skipped; set CI_GATES_FULL=1 for full).")

    text = open(DOC).read()

    # Retraction tripwire: scan the WHOLE document, not just section 0. A retracted claim that survives
    # outside the recomputed section is indistinguishable from a current one to a reader.
    # A retracted value may still APPEAR if the paragraph it sits in flags it as history -- documenting a
    # retraction is the opposite of hiding one. Asserting it as current is what must fail.
    OK_CONTEXT = ("retract", "earlier version", "superseded", "before the fix", "previously")

    def _para_of(txt, needle):
        i = txt.index(needle)
        start = txt.rfind("\n\n", 0, i) + 2
        end = txt.find("\n\n", i)
        return txt[start:end if end > 0 else len(txt)].lower()

    hits = []
    for val, why in facts["retracted_values"].items():
        if val in text:
            para = _para_of(text, val)
            if not any(marker in para for marker in OK_CONTEXT):
                hits.append((val, why))
    if hits:
        print("RETRACTED VALUES PRESENT IN THE DOCUMENT:")
        for val, why in hits:
            print(f"  {val!r}: {why}")
        print("Remove or rewrite these; a retraction that only reaches section 0 leaves the claim alive.")
        return 1

    block = section0(text)
    allowed = allowed_decimals(facts) | STRUCTURAL | grandfathered
    # Acceptance is EXACT at the token's own precision: a decimal with d places is fine
    # only if some computed value rounds to that token. Rounding the token first (the
    # obvious shortcut) would accept 66.98 because 66.99 and 66.98 share 1-dp form 67.0.
    allowed_floats = []
    for a in allowed:
        try:
            allowed_floats.append(float(a))
        except ValueError:
            pass

    def is_computed(tok):
        d = len(tok.split(".")[1])
        return any(f"{a:.{d}f}" == tok for a in allowed_floats)

    findings = []
    for lineno, line in enumerate(block.splitlines(), 1):
        for m in re.finditer(r"\b\d+\.\d+\b", line):
            tok = m.group(0)
            if is_computed(tok):
                continue
            findings.append((lineno, tok, line.strip()[:90]))

    nvals = len(set(a for a in allowed if re.match(r'^-?\d+\.\d+$', a)))
    print(f"section 0: {nvals} computed decimal values in the allowed set")
    if findings:
        print(f"\n{len(findings)} number(s) in section 0 that the current model does not produce:")
        for lineno, tok, ctx in findings:
            near = sorted(allowed, key=lambda a: abs(float(a) - float(tok)) if re.match(r'^\d+\.\d+$', a) else 1e9)[:3]
            print(f"  line {lineno}: {tok}  (nearest computed: {', '.join(near)})  :: {ctx}")
        return 1
    # gpu-sim-z1cw: measured-bytes paragraph must quote the committed measured values.
    _msb = facts.get("scaleout_opt", {}).get("measured_style_bytes", {})
    if _msb:
        _mpara = text[text.index("**Beyond one rack"):text.index("**CURRENT ANSWER")]
        _expected = []
        for _v in _msb.get("bf16_moments", {}).values():
            if isinstance(_v, (int, float)) and not isinstance(_v, bool):
                _expected.append(_v)
        for _v in _msb.get("fp32_moments", {}).values():
            if isinstance(_v, (int, float)) and not isinstance(_v, bool):
                _expected.append(_v)
        for _v in _msb.get("measured_floor_gb_405b_at_80", {}).values():
            if isinstance(_v, (int, float)) and not isinstance(_v, bool):
                _expected.append(_v)
        _pairs = [
            ("8.36", _msb["bf16_moments"]["fused"]),
            ("10.36", _msb["bf16_moments"]["foreach_true"]),
            ("4.36", _msb["bf16_moments"]["params_grads"]),
            ("16.36", _msb["fp32_moments"]["fused"]),
            ("20.36", _msb["fp32_moments"]["foreach_true"]),
            ("17.09", _msb["fp32_moments"]["foreach_false"]),
            ("83.1", _msb["measured_floor_gb_405b_at_80"]["fp32_moments_fused"]),
        ]
        _missing = []
        for _s, _v in _pairs:
            _vs = f"{_v:g}"
            if _s != _vs:
                _missing.append((f"facts value {_vs} != expected {_s}", _s))
                continue
            if _s not in _mpara:
                _missing.append((_v, _s))
        if _missing:
            print("\nmeasured-bytes paragraph is missing committed value(s):")
            for _v, _s in _missing:
                print(f"  {_s}  (from model_facts scaleout_opt.measured_style_bytes)")
            return 1
    print("OK: every decimal in section 0 is a value computed from the current model")
    return 0


if __name__ == "__main__":
    sys.exit(main())
