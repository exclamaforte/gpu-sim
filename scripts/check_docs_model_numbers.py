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

    facts = {"generated_by": "scripts/check_docs_model_numbers.py",
             "cp_convention_gap": _cp_gap,
             "sensitivity_radius": _json.loads(_rad.stdout),
             "model": "llama3-8b", "act_factor": ACT_FACTOR,
             "pricing_path": "spec/topology", "rank_by": "sum", "rows": {}}

    # gpu-sim-wv2 (partial): the compute term divides FLOPs by peak with no efficiency
    # factor. Record the implied MFU and the step at a measured (0.566) and pessimistic
    # (0.40) efficiency, so the doc can quote them and the numbers are reproducible here.
    facts["mfu_sensitivity"] = {}
    _implied = None
    for tag, mfu in (("implied_0.765", None), ("measured_5090", 0.566), ("typical_h100", 0.40)):
        peak = PEAK if mfu is None else PEAK * mfu / 0.765
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
            facts["mfu_sensitivity"][tag]["step_ms"] / facts["mfu_sensitivity"]["implied_0.765"]["step_ms"], 2)

    # gpu-sim-mfum: how DECISIVE is each winner? The margin against the best config with a
    # different (tp,pp,cp,dp) shape, and where the N=16 answer stops holding. A tie between
    # zero policies (same shape, zero=2 vs zero=3) is not a fragile win, so the runner-up is
    # taken over shapes, not over rows.
    def _winner(n, mfu):
        _, feas = search(n, peak=PEAK * mfu / 0.765)
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
                                             "implied 0.765 winner already has headroom)",
                                     "grid": {}}
    for n in (8, 16, 32):
        for seq in (8192, 16384, 32768):
            cell = {}
            for hr in (0.0, 0.10, 0.20):
                m = dict(M8)
                m["seq"] = seq
                hwv = {"peak_flops": PEAK * 0.566 / 0.765, "mem_gb": 80.0,
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
        hwv = {"peak_flops": PEAK * mfu / 0.765, "mem_gb": 80.0,
               "spec_path": "specs/h100_rack.json"}
        _, feas = ps.search(8, hwv, m)
        cp1 = [c for c in feas if c.cp == 1]
        cpg = [c for c in feas if c.cp > 1]
        if not cp1 or not cpg:
            return None
        return 100 * (min(c.step_sum for c in cp1) / min(c.step_sum for c in cpg) - 1)

    facts["cp_crossover"] = {}
    for tag, mfu in (("implied", 0.765), ("measured", 0.566), ("typical", 0.40)):
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
            for tag, mfu in (("implied", 0.765), ("measured", 0.566), ("typical", 0.40)):
                m = dict(M8)
                m["seq"] = seq
                hwv = {"peak_flops": PEAK * mfu / 0.765, "mem_gb": 80.0, "spec_path": SPEC[n]}
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
        for tag, mfu in (("implied", 0.765), ("measured", 0.566), ("typical_h100", 0.40)):
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
    s1 = text[text.index("**CURRENT ANSWER"):text.index("THE RANKING IS A NO-OVERLAP RANKING")]
    s2 = text[text.index("**Does the answer generalize past 8B?"):text.index("## 1. Premise")]
    return s1 + "\n" + s2


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
    for v in facts["cp_convention_gap"].values():
        add(v)
        add(round(v, 2))
    for row in facts["sensitivity_radius"].values():
        for key in ("flip_cp", "flip_pp"):
            if row.get(key) is not None:
                add(row[key])
                add(round(row[key], 2))
                add(round(row[key], 1))
    return vals


def main():
    facts = compute_facts()
    with open(FACTS_JSON, "w") as fh:
        json.dump(facts, fh, indent=2)

    grand = os.path.join(ROOT, "scripts/docs_model_numbers_grandfathered.txt")
    grandfathered = set()
    if os.path.exists(grand):
        for ln in open(grand):
            ln = ln.split("#")[0].strip()
            if ln:
                grandfathered.add(ln.split()[0])

    text = open(DOC).read()
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
    print("OK: every decimal in section 0 is a value computed from the current model")
    return 0


if __name__ == "__main__":
    sys.exit(main())
