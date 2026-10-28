#!/usr/bin/env python3
"""Error budget for the step predictor (gpu-sim-hd5u).

Ranks every unbounded parameter by how close a published verdict sits to its
flip radius, as a fraction of the parameter's plausible range. Reuses the
production model (epic_cluster.parallelism_search) via knobs; defaults are no-ops.

Usage: python3 scripts/error_budget.py [--json]
"""
from __future__ import annotations
import contextlib
import io
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

RECOMPUTE_PARAMS = {"recompute_charge", "alpha_scale"}

def hw_base(n, opt_state="fp32_master", recompute=0.0):
    h = {"peak_flops": PEAK, "mem_gb": 80.0, "spec_path": SPECS[n],
         "opt_state": opt_state, "opt_peak": "default", "seq": 8192,
         "recompute": recompute}
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
     "provenance": "measured (1.24-1.27x, gpu-sim-recompute)",
     "recompute": 1.0},
    {"name": "alpha_scale", "default": 1.0, "lo": 0.5, "hi": 1.684,
     "provenance": "measured on one RTX 5090 (0.176/0.420/0.594); effective range capped at 1/0.594",
     "recompute": 1.0},
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
]

EFF_PARAMS = ["eff_gemm", "eff_attention", "eff_layernorm", "eff_elementwise"]
EFF_MAP = {"eff_gemm": "gemm", "eff_attention": "attention",
           "eff_layernorm": "layernorm", "eff_elementwise": "elementwise"}
EFF_DEFAULTS = {"eff_gemm": 0.50, "eff_attention": 0.35,
                "eff_layernorm": 0.20, "eff_elementwise": 0.30}
EFF_RANGES = {"eff_gemm": (0.30, 0.70), "eff_attention": (0.20, 0.55),
              "eff_layernorm": (0.10, 0.40), "eff_elementwise": (0.15, 0.50)}

CELLS = [
    ("N8_winner", 8, 8192, "8b", "winner", None),
    ("N16_winner", 16, 8192, "8b", "winner", None),
    ("N32_winner", 32, 8192, "8b", "winner", None),
    ("N16_seq32768", 16, 32768, "8b", "winner", None),
    ("N32_seq32768", 32, 32768, "8b", "winner", None),
    ("405B_bf16_feasible", 80, 8192, "405b", "feasible", "bf16_moments"),
    ("405B_fp32master_feasible", 80, 8192, "405b", "feasible", "fp32_master"),
]

RECOMPUTE_CELLS = [
    ("N8_winner_recompute1", 8, 8192, "8b", "winner", None),
    ("N16_winner_recompute1", 16, 8192, "8b", "winner", None),
]

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
    results, _ = ps.search(N, h, m)
    best = min(results, key=lambda c: c.memory_gb)
    mem_limit = h["mem_gb"] * 1e9 * (1.0 - h.get("mem_headroom", 0.0))
    margin_gb = mem_limit / 1e9 - best.memory_gb
    return False, best, margin_gb

def recompute_for(param):
    return 1.0 if param in RECOMPUTE_PARAMS else 0.0

def verdict_for(param, value, cell):
    cname, N, seq, model_key, kind, opt_state = cell
    model = MODELS[model_key]
    hw = hw_base(N, opt_state=opt_state if opt_state else "fp32_master",
                 recompute=recompute_for(param))
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
    cname, N, seq, model_key, kind, opt_state = cell
    model = MODELS[model_key]
    hw = hw_base(N, opt_state=opt_state if opt_state else "fp32_master",
                 recompute=recompute_for(param))
    if opt_state:
        hw["opt_state"] = opt_state
    hw = set_param(hw, param, value)
    if kind == "winner":
        _, _, margin_ms = winner_cell(N, seq, model, hw)
        return margin_ms
    else:
        _, _, margin_gb = feas_cell(N, seq, model, hw, opt_state)
        return margin_gb

def winner_config(param, value, cell):
    """Return the winning config tuple for a winner cell at a given param value."""
    cname, N, seq, model_key, kind, opt_state = cell
    model = MODELS[model_key]
    hw = hw_base(N, opt_state=opt_state if opt_state else "fp32_master",
                 recompute=recompute_for(param))
    if opt_state:
        hw["opt_state"] = opt_state
    hw = set_param(hw, param, value)
    v, w, _ = winner_cell(N, seq, model, hw)
    return v

def bisect_flips(param, cell, default_val, lo, hi):
    """Return (flip_up, flip_down)."""
    base = verdict_for(param, default_val, cell)
    flip_up = None
    flip_down = None
    if verdict_for(param, hi, cell) != base:
        a, b = default_val, hi
        if a > b:
            a, b = b, a
        for _ in range(60):
            mid = (a + b) / 2
            if verdict_for(param, mid, cell) == base:
                a = mid
            else:
                b = mid
        flip_up = round(b, 6)
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
        flip_down = round(a, 6)
    return flip_up, flip_down

def elasticity(param, cell, default_val, lo, hi):
    step = (hi - lo) * 0.05
    if step == 0:
        return 0.0
    try:
        m0 = margin_for(param, default_val, cell)
        vp = min(hi, default_val + step)
        vm = max(lo, default_val - step)
        mp = margin_for(param, vp, cell)
        mm = margin_for(param, vm, cell)
        denom = vp - vm
        if denom == 0:
            return 0.0
        dmdx = (mp - mm) / denom
    except Exception:
        return 0.0
    if m0 == 0:
        return float("inf")
    return dmdx * (default_val / m0) if default_val != 0 else dmdx

def one_sided_fraction(default_val, flip, lo, hi):
    if flip is None:
        return None
    if flip < default_val:
        denom = default_val - lo
    else:
        denom = hi - default_val
    if denom == 0:
        return 1.0
    return abs(default_val - flip) / denom

def dominant_op(spec_path, eff_name=None, eff_value=None):
    base = dict(cy0_l1.EFF_CLASS)
    if eff_name is not None:
        base[EFF_MAP[eff_name]] = eff_value
    saved = dict(cy0_l1.EFF_CLASS)
    cy0_l1.EFF_CLASS.clear()
    cy0_l1.EFF_CLASS.update(base)
    try:
        res = cy0_l1.predict_layer(spec_path, tp=2)
    finally:
        cy0_l1.EFF_CLASS.clear()
        cy0_l1.EFF_CLASS.update(saved)
    comps = [c for c in res["components"] if c.get("kind") == "compute"]
    if not comps:
        return "none"
    return max(comps, key=lambda c: c["seconds"])["op"]

def compute():
    ranked = []
    for p in PARAMS:
        name = p["name"]
        lo, hi = p["lo"], p["hi"]
        default_val = p["default"]
        recompute = p.get("recompute", 0.0)
        cells = RECOMPUTE_CELLS if name in RECOMPUTE_PARAMS else CELLS
        for cell in cells:
            cname = cell[0]
            if name in RECOMPUTE_PARAMS:
                cname = cname + " (recompute=1.0, conditional)"
            base_verdict = verdict_for(name, default_val, cell)
            flip_up, flip_down = bisect_flips(name, cell, default_val, lo, hi)
            fw_candidates = [(x, abs(default_val - x) / (hi - lo) if (hi - lo) > 0 else 1.0)
                            for x in (flip_up, flip_down) if x is not None]
            if fw_candidates:
                flip_full = min(fw_candidates, key=lambda t: t[1])[0]
                frac = min(t[1] for t in fw_candidates)
            else:
                flip_full = None
                frac = None
            os_candidates = [(x, one_sided_fraction(default_val, x, lo, hi))
                            for x in (flip_up, flip_down) if x is not None]
            os_candidates = [(x, f) for x, f in os_candidates if f is not None]
            if os_candidates:
                flip_os = min(os_candidates, key=lambda t: t[1])[0]
                frac_os = min(t[1] for t in os_candidates)
            else:
                flip_os = None
                frac_os = None
            flip = flip_os
            if flip is None:
                frac = None
                frac_os = None
            if flip is None:
                m_lo = margin_for(name, lo, cell)
                m_hi = margin_for(name, hi, cell)
                # worst (smallest) margin across the range
                cands = [x for x in (m_lo, m_hi) if x is not None]
                margin = min(cands) if cands else None
            else:
                margin = margin_for(name, default_val, cell)
            elast = elasticity(name, cell, default_val, lo, hi)
            row = {
                "parameter": name,
                "affected_verdict": cname,
                "provenance": p["provenance"],
                "default": default_val,
                "range": [lo, hi],
                "base_verdict": [str(x) for x in base_verdict],
                "current_margin": margin,
                "margin_basis": ("at default" if flip is not None else "worst over range"),
                "flip_value": flip,
                "flip_up": flip_up,
                "flip_down": flip_down,
                "measured_sides": sorted((["up"] if flip_up is not None else []) + (["down"] if flip_down is not None else [])),
                "flip_full": flip_full,
                "flip_one_sided": flip_os,
                "fraction_of_range_consumed": (None if frac is None else round(frac, 6)),
                "fraction_one_sided": (None if frac_os is None else round(frac_os, 6)),
                # json.dumps writes inf/NaN as bare tokens, which are NOT valid JSON
                # (RFC 8259) and are rejected by jq / JSON.parse / serde. elasticity
                # is float('inf') when the verdict has ZERO current margin -- a real
                # and important case (any change flips it), so it is preserved as an
                # explicit flag rather than silently nulled.
                "elasticity": (None if (elast is None or (isinstance(elast, float) and (math.isnan(elast) or math.isinf(elast)))) else round(elast, 4)),
                "elasticity_infinite": bool(isinstance(elast, float) and math.isinf(elast)),
            }
            # bracketing pair
            if flip is not None and cname.startswith("N") and "winner" in cname:
                try:
                    below = winner_config(name, flip * 0.999, cell) if flip > default_val else winner_config(name, default_val, cell)
                    above = winner_config(name, flip * 1.001, cell) if flip > default_val else winner_config(name, flip * 0.999, cell)
                    row["bracket_below"] = [str(x) for x in below]
                    row["bracket_above"] = [str(x) for x in above]
                except Exception:
                    pass
            ranked.append(row)

    def _sort_key(r):
        f = r["fraction_one_sided"]
        if f is None:
            cm = r.get("current_margin")
            cm = 0.0 if cm is None else cm
            return (1, cm, 0.0)
        # Infinite elasticity sorts FIRST (most sensitive) -- which is what
        # -inf did before the value was made JSON-safe.
        if r.get("elasticity_infinite"):
            return (0, f, float("-inf"))
        return (0, f, -(abs(r["elasticity"]) if r["elasticity"] is not None else 0))
    ranked.sort(key=_sort_key)

    # unranked (no discrete published verdict)
    unranked = []
    for name in EFF_PARAMS:
        lo, hi = EFF_RANGES[name]
        default_val = EFF_DEFAULTS[name]
        base_verdict = dominant_op("specs/h100_rack.json", name, default_val)
        # try flip in both directions
        flip = None
        if dominant_op("specs/h100_rack.json", name, hi) != base_verdict:
            a, b = default_val, hi
            for _ in range(60):
                mid = (a + b) / 2
                if dominant_op("specs/h100_rack.json", name, mid) == base_verdict:
                    a = mid
                else:
                    b = mid
            flip = round(b, 6)
        elif dominant_op("specs/h100_rack.json", name, lo) != base_verdict:
            a, b = lo, default_val
            for _ in range(60):
                mid = (a + b) / 2
                if dominant_op("specs/h100_rack.json", name, mid) == base_verdict:
                    b = mid
                else:
                    a = mid
            flip = round(a, 6)
        if flip is None:
            frac = None
            frac_os = None
        else:
            frac = abs(default_val - flip) / (hi - lo) if (hi - lo) > 0 else 1.0
            frac_os = one_sided_fraction(default_val, flip, lo, hi)
        unranked.append({
            "parameter": name,
            "affected_verdict": "cy0 single-layer dominant op (not a published verdict)",
            "provenance": "published-spec",
            "default": default_val,
            "range": [lo, hi],
            "base_verdict": [base_verdict],
            "current_margin": None,
            "flip_value": flip,
            "fraction_of_range_consumed": (None if frac is None else round(frac, 6)),
            "fraction_one_sided": (None if frac_os is None else round(frac_os, 6)),
            "measured_sides": [],
            "elasticity": None,
            "ranked": False,
        })

    result = {"ranked": ranked, "unranked": unranked}

    return ranked, unranked, result

def main():
    ranked, unranked, result = compute()

    if "--json" in sys.argv:
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    print(f"{'parameter':24s} {'verdict':32s} {'frac':>8s} {'1side':>8s} {'flip':>10s} {'margin':>12s}")
    for r in ranked:
        f = r['fraction_of_range_consumed']
        o = r['fraction_one_sided']
        fs = f"{f:8.4f}" if f is not None else "     n/a"
        os_ = f"{o:8.4f}" if o is not None else "     n/a"
        print(f"{r['parameter']:24s} {r['affected_verdict']:32s} {fs} {os_} {str(r['flip_value']):>10s} {str(r['current_margin']):>12s}")
    print()
    print("Unranked (no discrete published verdict):")
    for r in unranked:
        print(f"  {r['parameter']:24s} {r['affected_verdict']}")
    print()
    if ranked:
        top = ranked[0]
        print(f"TOP: {top['parameter']} on {top['affected_verdict']} (fraction {top['fraction_of_range_consumed']})")
    # bracketing pairs
    print()
    print("Bracketing pairs (config below flip -> config above flip):")
    for r in ranked:
        if "bracket_below" in r:
            print(f"  {r['parameter']} @ {r['flip_value']} on {r['affected_verdict']}: "
                  f"{r['bracket_below']} -> {r['bracket_above']}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
