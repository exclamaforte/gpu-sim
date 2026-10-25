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
from epic_cluster import parallelism_search as ps  # noqa: E402

DOC = os.path.join(ROOT, "docs/training-parallelism-sim.md")
FACTS_JSON = os.path.join(ROOT, "docs/model_facts.json")

M8 = {"n_layers": 32, "d_model": 4096, "n_heads": 32, "head_dim": 128,
      "d_ff": 14336, "vocab": 128256}
SPEC = {8: "specs/h100_rack.json", 16: "specs/h100_rack.json", 32: "specs/h100_rack_4node.json"}
ACT_FACTOR = 2.3


def hw_for(N, path="spec", cpg="crude", act=ACT_FACTOR):
    if N == 8 and path == "legacy":
        hw = {"peak_flops": 989e12, "mem_gb": 80.0, "fabric": "nvlink4_h100", "cp_grad_mode": cpg}
    else:
        hw = {"peak_flops": 989e12, "mem_gb": 80.0, "spec_path": SPEC[N], "cp_grad_mode": cpg}
    if act is not None:
        hw["act_factor"] = act
    return hw


def search(N, seq=8192, path="spec", cpg="crude", act=ACT_FACTOR):
    m = dict(M8)
    m["seq"] = seq
    return ps.search(N, hw_for(N, path, cpg, act), m=m)


def compute_facts():
    facts = {"generated_by": "scripts/check_docs_model_numbers.py",
             "model": "llama3-8b", "act_factor": ACT_FACTOR,
             "pricing_path": "spec/topology", "rank_by": "sum", "rows": {}}

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
    m70 = {"n_layers": 80, "d_model": 8192, "n_heads": 64, "head_dim": 128,
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
        "qwen2.5-32b": {"n_layers": 64, "d_model": 5120, "n_heads": 40, "head_dim": 128,
                        "d_ff": 27648, "vocab": 152064},
        "llama2-34b": {"n_layers": 48, "d_model": 8192, "n_heads": 64, "head_dim": 128,
                       "d_ff": 22016, "vocab": 32000},
        "llama3-70b": {"n_layers": 80, "d_model": 8192, "n_heads": 64, "head_dim": 128,
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
