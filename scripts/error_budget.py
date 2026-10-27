#!/usr/bin/env python3
"""Error budget for the step predictor (gpu-sim-hd5u).

Ranks every unbounded parameter by how close a published verdict sits to its
flip radius, as a fraction of the parameter's plausible range. Reuses the
production model (epic_cluster.parallelism_search) via knobs; defaults are no-ops.

Usage: python3 scripts/error_budget.py [--json]
"""
from __future__ import annotations
import json, math, sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from epic_cluster import parallelism_search as ps
import cy0_l1

PEAK = 989e12
SPECS = {8: "specs/h100_rack.json", 16: "specs/h100_rack.json",
         32: "specs/h100_rack_4node.json", 80: "experiments/gpu-sim-hd5u/spec_10node.json"}
MODELS = {
    "8b": {"n_layers": 32, "d_model": 4096, "n_heads": 32, "n_kv_heads": 8, "head_dim": 128,
           "d_ff": 14336, "vocab": 128256, "seq": 8192},
    "405b": {"n_layers": 126, "d_model": 16384, "n_heads": 128, "n_kv_heads": 8, "head_dim": 128,
             "d_ff": 53248, "vocab": 128256, "seq": 8192},
}

def hw_base(n, opt_state="fp32_master"):
    h = {"peak_flops": PEAK, "mem_gb": 80.0, "spec_path": SPECS[n],
         "opt_state": opt_state, "opt_peak": "default", "seq": 8192}
    return h

def set_param(hw, name, value):
    hw = json.loads(json.dumps(hw))
    if name == "term_scale_cp":
        hw.setdefault("term_scale", {})["cp"] = value
    elif name == "term_scale_pp":
        hw.setdefault("term_scale", {})["pp"] = value
    elif name == "recompute":
        hw["recompute"] = value
    elif name == "recompute_charge":
        hw["recompute_charge"] = value
    elif name == "alpha_scale":
        hw["alpha_scale"] = value
    elif name == "microbatch_factor":
        hw["microbatch_factor"] = value
    elif name == "act_factor":
        hw["act_factor"] = value
    elif name == "achieved_bw_scale":
        hw["achieved_bw_scale"] = value
    elif name == "mem_headroom":
        hw["mem_headroom"] = value
    elif name == "mem_charge":
        hw["mem_charge"] = value
    return hw

PARAMS = [
    {"name": "term_scale_cp", "default": 1.0, "lo": 0.25, "hi": 4.0,
     "provenance": "assumed (measured 4x error once, gpu-sim-kvcp)"},
    {"name": "term_scale_pp", "default": 1.0, "lo": 0.25, "hi": 4.0,
     "provenance": "assumed (no measurement)"},
    {"name": "recompute", "default": 0.0, "lo": 0.0, "hi": 1.0,
     "provenance": "measured (1.24-1.27x, gpu-sim-recompute)"},
    {"name": "recompute_charge", "default": 0.25, "lo": 0.20, "hi": 0.35,
     "provenance": "measured (1.24-1.27x, gpu-sim-recompute)"},
    {"name": "alpha_scale", "default": 1.0, "lo": 0.5, "hi": 2.0,
     "provenance": "measured on one RTX 5090 (0.176/0.420/0.594)"},
    {"name": "microbatch_factor", "default": 1.0, "lo": 1.0, "hi": 8.0,
     "provenance": "assumed (no measurement)"},
    {"name": "act_factor", "default": 2.3, "lo": 1.0, "hi": 4.0,
     "provenance": "measured (gpu-sim-4nf, 2.3x)"},
    {"name": "achieved_bw_scale", "default": 1.0, "lo": 0.5, "hi": 2.0,
     "provenance": "published-spec (datasheet, not measured here)"},
    {"name": "mem_headroom", "default": 0.0, "lo": 0.0, "hi": 0.2,
     "provenance": "assumed (no measurement)"},
    {"name": "mem_charge", "default": 1.0, "lo": 1.0, "hi": 1.25,
     "provenance": "measured (scup 1.25x memory charge)"},
    # cy0 eff classes: no discrete verdict in parallelism_search; included for coverage
    {"name": "eff_gemm", "default": 0.50, "lo": 0.30, "hi": 0.70,
     "provenance": "published-spec (H100 whitepaper/Megatron)"},
    {"name": "eff_attention", "default": 0.35, "lo": 0.20, "hi": 0.55,
     "provenance": "published-spec"},
    {"name": "eff_layernorm", "default": 0.20, "lo": 0.10, "hi": 0.40,
     "provenance": "published-spec"},
    {"name": "eff_elementwise", "default": 0.30, "lo": 0.15, "hi": 0.50,
     "provenance": "published-spec"},
]

CY0_NAMES = {"eff_gemm", "eff_attention", "eff_layernorm", "eff_elementwise"}
EFF_MAP = {"eff_gemm": "gemm", "eff_attention": "attention",
           "eff_layernorm": "layernorm", "eff_elementwise": "elementwise"}

def winner_cell(N, seq, model, hw):
    m = dict(model, seq=seq)
    h = dict(hw, seq=seq)
    _, feas = ps.search(N, h, m)
    if not feas:
        return ("INFEASIBLE",), None, None
    w = feas[0]
    margin_ms = (feas[1].step_sum - w.step_sum) * 1000 if len(feas) > 1 else float("inf")
    return tuple([w.tp, w.pp, w.cp, w.dp, w.zero]), w, margin_ms

def feas_cell(N, seq, model, hw, opt_state):
    m = dict(model, seq=seq)
    h = dict(hw, seq=seq, opt_state=opt_state)
    _, feas = ps.search(N, h, m)
    if feas:
        w = feas[0]
        mem_limit = h["mem_gb"] * 1e9 * (1.0 - h.get("mem_headroom", 0.0))
        margin_gb = mem_limit / 1e9 - w.memory_gb
        return True, w, margin_gb
    # no feasible: use lowest-memory config
    results, _ = ps.search(N, h, m)
    best = min(results, key=lambda c: c.memory_gb)
    mem_limit = h["mem_gb"] * 1e9 * (1.0 - h.get("mem_headroom", 0.0))
    margin_gb = mem_limit / 1e9 - best.memory_gb
    return False, best, margin_gb

def verdict_for(param, value, cell):
    _cname, N, seq, model_key, kind, opt_state = cell
    model = MODELS[model_key]
    hw = hw_base(N)
    if opt_state:
        hw["opt_state"] = opt_state
    hw = set_param(hw, param, value)
    if kind == "winner":
        v, _, _ = winner_cell(N, seq, model, hw)
        return v
    else:
        flag, _, _ = feas_cell(N, seq, model, hw, opt_state)
        return (flag,)

def margin_for(param, value, cell):
    _cname, N, seq, model_key, kind, opt_state = cell
    model = MODELS[model_key]
    hw = hw_base(N)
    if opt_state:
        hw["opt_state"] = opt_state
    hw = set_param(hw, param, value)
    if kind == "winner":
        _, _, margin_ms = winner_cell(N, seq, model, hw)
        return margin_ms
    else:
        _, _, margin_gb = feas_cell(N, seq, model, hw, opt_state)
        return margin_gb

CELLS = [
    ("N8_winner", 8, 8192, "8b", "winner", None),
    ("N16_winner", 16, 8192, "8b", "winner", None),
    ("N32_winner", 32, 8192, "8b", "winner", None),
    ("405B_bf16_feasible", 80, 8192, "405b", "feasible", "bf16_moments"),
    ("405B_fp32master_feasible", 80, 8192, "405b", "feasible", "fp32_master"),
]

def bisect_flip(param, cell, default_val, lo, hi):
    base = verdict_for(param, default_val, cell)
    # upward
    if verdict_for(param, hi, cell) != base:
        a, b = default_val, hi
        # ensure a<b
        if a > b:
            a, b = b, a
        for _ in range(60):
            mid = (a + b) / 2
            if verdict_for(param, mid, cell) == base:
                a = mid
            else:
                b = mid
        return round(b, 6)
    # downward
    if verdict_for(param, lo, cell) != base:
        a, b = lo, default_val
        if a > b:
            a, b = b, a
        for _ in range(60):
            mid = (a + b) / 2
            if verdict_for(param, mid, cell) == base:
                b = mid
            else:
                a = mid
        return round(a, 6)
    return None

def elasticity(param, cell, default_val, lo, hi):
    eps = (hi - lo) * 1e-4
    if eps == 0:
        return 0.0
    try:
        m0 = margin_for(param, default_val, cell)
        mp = margin_for(param, min(hi, default_val + eps), cell)
        mm = margin_for(param, max(lo, default_val - eps), cell)
        denom = (min(hi, default_val + eps) - max(lo, default_val - eps))
        if denom == 0:
            return 0.0
        dmdx = (mp - mm) / denom
    except Exception:
        return 0.0
    if m0 == 0:
        return float("inf")
    return dmdx * (default_val / m0)


def dominant_op(spec_path, eff_name=None, eff_value=None):
    import copy
    base = dict(cy0_l1.EFF_CLASS)
    if eff_name is not None:
        base[EFF_MAP[eff_name]] = eff_value
    cy0_l1.EFF_CLASS.clear()
    cy0_l1.EFF_CLASS.update(base)
    res = cy0_l1.predict_layer(spec_path, tp=2)
    comps = [c for c in res["components"] if c.get("kind") == "compute"]
    if not comps:
        return "none"
    return max(comps, key=lambda c: c["seconds"])["op"]

def main():
    rows = []
    for p in PARAMS:
        name = p["name"]
        if name in CY0_NAMES:
            spec_path = "specs/h100_rack.json"
            base_verdict = dominant_op(spec_path, name, p["default"])
            # bisect flip (linear, since eff in [0,1])
            flip = None
            lo, hi = p["lo"], p["hi"]
            if dominant_op(spec_path, name, hi) != base_verdict:
                a, b = p["default"], hi
                for _ in range(60):
                    mid = (a+b)/2
                    if dominant_op(spec_path, name, mid) == base_verdict:
                        a = mid
                    else:
                        b = mid
                flip = round(b, 6)
            elif dominant_op(spec_path, name, lo) != base_verdict:
                a, b = lo, p["default"]
                for _ in range(60):
                    mid = (a+b)/2
                    if dominant_op(spec_path, name, mid) == base_verdict:
                        b = mid
                    else:
                        a = mid
                flip = round(a, 6)
            if flip is None:
                frac = 1.0
            else:
                frac = abs(p["default"] - flip) / (hi - lo) if (hi-lo)>0 else 1.0
            rows.append({
                "parameter": name,
                "affected_verdict": "cy0_dominant_op",
                "provenance": p["provenance"],
                "default": p["default"],
                "range": [p["lo"], p["hi"]],
                "base_verdict": [base_verdict],
                "current_margin": None,
                "flip_value": flip,
                "fraction_of_range_consumed": round(frac, 6),
                "elasticity": None,
            })
            continue
        for cell in CELLS:
            cname = cell[0]
            base_verdict = verdict_for(name, p["default"], cell)
            flip = bisect_flip(name, cell, p["default"], p["lo"], p["hi"])
            if flip is None:
                frac = 1.0
            else:
                frac = abs(p["default"] - flip) / (p["hi"] - p["lo"]) if (p["hi"] - p["lo"]) > 0 else 1.0
            margin = margin_for(name, p["default"], cell)
            elast = elasticity(name, cell, p["default"], p["lo"], p["hi"])
            rows.append({
                "parameter": name,
                "affected_verdict": cname,
                "provenance": p["provenance"],
                "default": p["default"],
                "range": [p["lo"], p["hi"]],
                "base_verdict": [str(x) for x in base_verdict],
                "current_margin": margin,
                "flip_value": flip,
                "fraction_of_range_consumed": round(frac, 6),
                "elasticity": (None if (isinstance(elast, float) and math.isnan(elast)) else round(elast, 4)),
            })
    # rank by fraction ascending (most sensitive first), then elasticity desc
    rows.sort(key=lambda r: (r["fraction_of_range_consumed"], -(abs(r["elasticity"]) if r["elasticity"] is not None else 0)))
    if "--json" in sys.argv:
        print(json.dumps(rows, indent=2, sort_keys=True))
        return 0
    print(f"{'parameter':24s} {'verdict':30s} {'frac':>8s} {'flip':>10s} {'margin':>12s}")
    for r in rows:
        print(f"{r['parameter']:24s} {r['affected_verdict']:30s} {r['fraction_of_range_consumed']:8.4f} {str(r['flip_value']):>10s} {str(r['current_margin']):>12s}")
    print()
    if rows:
        top = rows[0]
        print(f"TOP: {top['parameter']} on {top['affected_verdict']} (fraction {top['fraction_of_range_consumed']})")
    return 0

if __name__ == "__main__":
    sys.exit(main())
