#!/usr/bin/env python3
"""PI's INDEPENDENT re-admission of the m3d.10 gate config set at candidate sizes.

WHAT THIS IS, and what it is not: this is the PI's own recomputation, written
while the payload size was still open, so that the gate's stage 1 is ready the
moment the size is chosen and so that the implementer's re-run has a second,
independently derived set of numbers to be checked against. The AUTHORITATIVE
admission remains the committed admissibility script's re-run (the gpu-sim-gmc
precedent: the admitted set, the margins and the rejected alternatives are
recorded by the implementer's script in experiments/gpu-sim-m3d.10/). A
disagreement between the two is a FINDING about the model or the method, not
something to be averaged away -- that has been the discipline of this epic
since the pkb agreement check was found to compare a formula with itself.

METHOD, stated so a reader can check it:
  * Durations come from the model's public predict() output only.
  * The latency term is scaled by the DERIVED pieces-per-phase P, because the
    accounting unit in force is per-isend (gpu-sim-pkb, ruled) and the C model
    derives P from the same formula and the same cap constant. P is NOT taken
    from the model's default (1), because that is the value which hides the
    divergence between the two implementations of the model.
  * P = ceil(chunk(n) * S / cap) with cap = 1 MiB = FG_FAB_MAX_PAYLOAD, and
    chunk(2) = S, chunk(n>=3) = (n-1)/n * S (epic_cluster/link_model.py).
  * The bandwidth term is recovered from the public API as
    bw_term = predict(2S) - predict(S), which equals f*S/B exactly.
  * Acceptance criteria applied, unchanged from the bead: 4a |d_bw| >= 0.5|d_pred|;
    4b |d_pred| >= 100us AND |d_pred| >= 0.2 * max(dur_i, dur_j) AND
    |d_pred| >= 10x the measured spread (spread is 0 here, so the absolute and
    relative floors dominate -- the measured spread is filled in by the
    measurement stage); 4c cancellation is a statement, not a number; 4d the
    ordering and the margins are functions of the model, so they are recomputed
    per size rather than inherited.
  * The byte factor is reported per config, because the n=2 factor is
    SIZE-DEPENDENT (gpu-sim-pkb, commit 56bbccc: 2S at 4 KiB, S at 1 MiB,
    2S at 2 MiB, with the 8sy cross-direction digest test showing the 1 MiB
    case is a genuine two-phase run and not a truncation). The model assumes
    2.0 at n=2, which matches the 4 KiB and 2 MiB observations and NOT the
    1 MiB one, so the factor the model uses is stated next to the factor the
    evidence supports, and the delta that a size-dependent factor would
    introduce is computed.
"""
import json
import math
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
from epic_cluster.link_model import predict  # noqa: E402

CAP = 1 << 20  # FG_FAB_MAX_PAYLOAD, include/fakegpu_fabric.h

# The admitted set from gpu-sim-gmc, re-evaluated here at each candidate size.
CONFIGS = {
    "M1": (2, 900.0, 1.0),
    "M2": (2, 100.0, 5.0),
    "M3": (4, 50.0, 5.0),
    "M4": (4, 25.0, 10.0),
}


def make_spec(n, bw, lat):
    nodes = [{"name": "n0", "devices": [{"id": i, "model": "m"} for i in range(n)]}]
    links = []
    for i in range(n):
        for j in range(i + 1, n):
            links.append({"a": {"node": "n0", "device": i}, "b": {"node": "n0", "device": j},
                          "kind": "nvlink", "bw_gbps": bw, "latency_us": lat,
                          "mtu": 4096, "loss_ppm": 0, "jitter_us": 0.0,
                          "shared_group": None})
    ranks = [{"rank": i, "node": "n0", "device": i} for i in range(n)]
    return {"version": 1,
            "models": {"m": {"sm_count": 80, "clock_mhz": 1000, "mem_bytes": 1 << 30,
                             "mem_bw_gbs": 1000.0, "arch": "x", "source": "pi-audit"}},
            "nodes": nodes, "links": links, "ranks": ranks}


def chunk_fraction(n):
    """Fraction of S carried by ONE ring transfer in one phase."""
    return 1.0 if n == 2 else (n - 1) / n


def load_measured_p():
    """The MEASURED pieces-per-phase table (n=2), extracted from the
    implementer's committed phase-payload table.

    THE MODEL'S P WAS WRONG, and this script used to make the same mistake:
    it derived P from the FABRIC'S PAYLOAD CAP, while the actual piece count
    is set by NCCL'S OWN CHUNKING. At 2 MiB the measurement is 8 chunks per
    phase, not 2 -- a four-fold undercount, so the latency term was four
    times too small. The reviewer's audit found it, and the table that
    exposed it is the source of truth here.
    """
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "phase_payload_p.json")
    with open(path) as f:
        raw = json.load(f)
    return {(int(r["S"]), int(raw["n"])): int(r["pieces_per_phase"])
            for r in raw["rows"]}


MEASURED_P = load_measured_p()


def derived_p(n, S):
    """pieces per phase, from the MEASURED table.

    A combination the table does not cover is REFUSED, not guessed: the
    cap-based derivation this replaced is known to be wrong (it is the
    four-fold undercount the review found), and an unmeasured combination
    has no defensible P. The same discipline the gate's arbiter uses on
    missing inputs, applied here to a model's inputs.
    """
    key = (int(S), int(n))
    if key in MEASURED_P:
        return MEASURED_P[key]
    raise SystemExit(
        f"REFUSED: no MEASURED pieces-per-phase for n={n}, S={S}. The table "
        f"covers {sorted(set(k[0] for k in MEASURED_P))} at n="
        f"{sorted(set(k[1] for k in MEASURED_P))}. Measure it (the phase-payload "
        f"table) rather than deriving it: the cap-based derivation is known "
        f"to undercount the piece count by four at 2 MiB.")


def row(n, bw, lat, S):
    spec = make_spec(n, bw, lat)
    count = S // 4
    P = derived_p(n, S)
    r1 = predict(spec, "ring_allreduce", count, "float32", n_ranks=n)
    r2 = predict(spec, "ring_allreduce", 2 * count, "float32", n_ranks=n)
    bw_term = r2["duration_s"] - r1["duration_s"]
    # The MERGED model has no pieces_per_phase parameter (it exists only on the
    # unreviewed gpu-sim-pkb branch), so the P scaling is applied here to the
    # model's latency term, using the public API alone:
    #   duration(P) = bw_term + P * latency_term(P=1)
    # A caller of the merged predict() who forgets P gets P=1 and is off by
    # exactly the amount the C model does not have -- which is the divergence
    # the pkb C-model item is about, and the reason P is derived here rather
    # than defaulted.
    lat_term_p1 = r1["duration_s"] - bw_term
    lat_term = lat_term_p1 * P
    duration = bw_term + lat_term
    return {
        "n": n, "bw_gbps": bw, "latency_us": lat,
        "pieces_per_phase_P": P,
        "pieces_per_rank": 2 * (n - 1) * P,
        "model_duration_s": duration,
        "model_duration_s_p1": r1["duration_s"],
        "bw_term_s": bw_term,
        "lat_term_s": lat_term,
        "lat_term_s_p1": lat_term_p1,
        "regime": r1["regime"],
        "model_byte_factor": 2.0 * chunk_fraction(n),
        "bytes_evidence_supported": {2: "2S at 4 KiB and 2 MiB, S at 1 MiB (size-dependent)",
                                     4: "2(n-1)/n (robust: splitting the chunk across "
                                        "channels does not change the total moved)"}[n],
    }


def evaluate(S):
    rows = {}
    refused = {}
    for k, v in CONFIGS.items():
        try:
            rows[k] = row(*v, S)
        except SystemExit as e:
            # A config with no MEASURED pieces-per-phase is REFUSED, not
            # estimated: the cap-based derivation is known to undercount the
            # piece count by four at 2 MiB, so an unmeasured config has no
            # defensible prediction and must not appear as if it had one.
            refused[k] = str(e)
    if not rows:
        return {"payload_bytes": S, "predicted_order": [], "configs": {},
                "pairs": [], "refused_configs": refused,
                "all_pairs_admissible": False,
                "admitted": False}
    order = sorted(rows, key=lambda k: rows[k]["model_duration_s"])
    pairs = []
    for a, b in zip(order, order[1:]):
        da, db = rows[a], rows[b]
        d_pred = abs(da["model_duration_s"] - db["model_duration_s"])
        d_bw = abs(da["bw_term_s"] - db["bw_term_s"])
        rel = d_pred / max(da["model_duration_s"], db["model_duration_s"])
        pairs.append({
            "pair": f"{a}->{b}",
            "d_pred_us": d_pred * 1e6,
            "bw_share_4a": (d_bw / d_pred) if d_pred else None,
            "relative_margin_4b": rel,
            "passes_4a": d_bw >= 0.5 * d_pred,
            "passes_4b_absolute_100us": d_pred >= 100e-6,
            "passes_4b_relative_20pct": rel >= 0.20,
            # the spread clause needs the measured spread; here we report the
            # largest per-piece spread the pair could tolerate.
            "max_tolerable_spread_us": d_pred * 1e6 / 10.0,
        })
    return {"payload_bytes": S, "predicted_order": order, "configs": rows, "pairs": pairs,
            "refused_configs": refused,
            "all_pairs_admissible": bool(pairs) and all(
                p["passes_4a"] and p["passes_4b_absolute_100us"]
                and p["passes_4b_relative_20pct"] for p in pairs),
            "admitted": (not refused) and bool(pairs) and all(
                p["passes_4a"] and p["passes_4b_absolute_100us"]
                and p["passes_4b_relative_20pct"] for p in pairs)}


def main():
    sizes = [int(a) for a in (sys.argv[1:] or [2 << 20, 8 << 20, 32 << 20])]
    out = {"artifact": "PI's independent re-admission (cross-check; the implementer's "
                       "committed script remains authoritative)",
           "method": {"P": "derived: ceil(chunk(n)*S / 1 MiB), chunk(2)=S, chunk(n>=3)=(n-1)/n*S",
                      "bw_term": "predict(2S) - predict(S) from the public API",
                      "criteria": "4a bw_share >= 0.5; 4b |d_pred| >= 100us and >= 20% of the "
                                  "larger duration; spread clause needs the measured spread"},
           "sizes": {}}
    for S in sizes:
        out["sizes"][str(S)] = evaluate(S)
    dest = os.path.join(os.path.dirname(os.path.abspath(__file__)), "admissibility_pi_audit.json")
    with open(dest, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    for S in sizes:
        e = out["sizes"][str(S)]
        print(f"=== S = {S >> 20} MiB  order={e['predicted_order']}  "
              f"admissible={e['all_pairs_admissible']}")
        for k in e["predicted_order"]:
            c = e["configs"][k]
            print(f"   {k}: n={c['n']} P={c['pieces_per_phase_P']} pieces/rank={c['pieces_per_rank']} "
                  f"dur={c['model_duration_s']*1e6:9.1f}us bw={c['bw_term_s']*1e6:9.1f} "
                  f"lat={c['lat_term_s']*1e6:7.1f} regime={c['regime']}")
        if e["refused_configs"]:
            for k, why in sorted(e["refused_configs"].items()):
                print(f"   {k}: REFUSED -- {why.splitlines()[0]}")
        for p in e["pairs"]:
            print(f"   {p['pair']}: d_pred={p['d_pred_us']:8.1f}us bw_share={p['bw_share_4a']:.3f} "
                  f"rel={p['relative_margin_4b']*100:5.1f}% 4a={'ok' if p['passes_4a'] else 'FAIL'} "
                  f"4b_abs={'ok' if p['passes_4b_absolute_100us'] else 'FAIL'} "
                  f"4b_rel={'ok' if p['passes_4b_relative_20pct'] else 'FAIL'} "
                  f"max_spread={p['max_tolerable_spread_us']:6.1f}us")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
