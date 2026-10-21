#!/usr/bin/env python3
"""m3d.10 item 4: pre-run admissibility check for a config set.

Computed entirely from the model's public predict() output. A config set is
ADMITTED only if, for every adjacent pair in the model's predicted order:
  4a bw-sensitivity: |d_bw| >= 0.5 * |d_pred|
  4b resolvability:  |d_pred| >= 100us, |d_pred| >= 0.2 * max(dur_i, dur_j),
                     and |d_pred| >= 10x measured spread (spread assumed 0 here,
                     so the absolute/relative floors dominate)
  4d ordering recorded (model-dependent).
"""
import sys, json, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from epic_cluster.link_model import predict

SIZE_BYTES = 32 * 1024 * 1024
COUNT = SIZE_BYTES // 4


def make_spec(n, bw, lat):
    nodes = [{"name": "n0", "devices": [{"id": i, "model": "m"} for i in range(n)]}]
    links = []
    for i in range(n):
        for j in range(i + 1, n):
            links.append({"a": {"node": "n0", "device": i}, "b": {"node": "n0", "device": j},
                          "kind": "nvlink", "bw_gbps": bw, "latency_us": lat,
                          "mtu": 4096, "loss_ppm": 0, "jitter_us": 0.0, "shared_group": None})
    ranks = [{"rank": i, "node": "n0", "device": i} for i in range(n)]
    return {"version": 1, "models": {"m": {"sm_count": 80, "clock_mhz": 1000, "mem_bytes": 1 << 30,
                                           "mem_bw_gbs": 1000.0, "arch": "x", "source": "test"}},
            "nodes": nodes, "links": links, "ranks": ranks}


CONFIGS = {
    "A": (2, 900.0, 1.0),
    "B": (2, 32.0, 5.0),
    "C": (4, 900.0, 1.0),
    "D": (4, 25.0, 50.0),
}


def bw_term(spec, count):
    # bw_term = predict(2S) - predict(S)  (== f*S/B)
    r1 = predict(spec, "ring_allreduce", count, "float32", n_ranks=spec and len(spec["ranks"]))
    r2 = predict(spec, "ring_allreduce", 2 * count, "float32", n_ranks=len(spec["ranks"]))
    return r2["duration_s"] - r1["duration_s"]


def eval_set(names):
    rows = {}
    for name in names:
        n, bw, lat = CONFIGS[name]
        spec = make_spec(n, bw, lat)
        r = predict(spec, "ring_allreduce", COUNT, "float32", n_ranks=n)
        bt = bw_term(spec, COUNT)
        rows[name] = {"n": n, "bw_gbps": bw, "latency_us": lat,
                      "predicted_s": r["duration_s"], "regime": r["regime"],
                      "bw_term": bt, "latency_term": r["duration_s"] - bt}
    # predicted order
    order = sorted(rows, key=lambda k: rows[k]["predicted_s"])
    pairs = []
    admitted = True
    reasons = []
    for i in range(len(order) - 1):
        a, b = order[i], order[i + 1]
        d_pred = rows[b]["predicted_s"] - rows[a]["predicted_s"]
        d_bw = rows[b]["bw_term"] - rows[a]["bw_term"]
        rel_margin = abs(d_pred) / max(rows[a]["predicted_s"], rows[b]["predicted_s"])
        bw_ratio = (abs(d_bw) / abs(d_pred)) if d_pred else 0.0
        ok_bw = bw_ratio >= 0.5
        ok_res = (abs(d_pred) >= 100e-6 and rel_margin >= 0.20)
        ok = ok_bw and ok_res
        if not ok:
            admitted = False
        pair = {"pair": f"{a}->{b}", "d_pred_us": d_pred * 1e6,
                "d_bw_us": d_bw * 1e6, "bw_ratio": bw_ratio,
                "rel_margin": rel_margin, "admissible": ok}
        if not ok_bw:
            pair["reason"] = "bw-sensitivity (4a) fails"
        if not ok_res:
            pair["reason"] = pair.get("reason", "") + "; resolvability (4b) fails"
        pairs.append(pair)
    return {"configs": names, "order": order, "rows": rows, "pairs": pairs,
            "admitted": admitted}


def main():
    candidate_sets = {
        "original_ABCD": ["A", "B", "C", "D"],
        "link_class_ladder_n2": None,  # placeholder, handled below
    }
    # link-class ladder: all n=2 at different bw
    # build as custom
    results = {}
    results["original_ABCD"] = eval_set(["A", "B", "C", "D"])

    # link-class ladder all n=2
    ladder_configs = {"L1": (2, 900.0, 1.0), "L2": (2, 400.0, 2.0),
                      "L3": (2, 100.0, 5.0), "L4": (2, 25.0, 10.0)}
    # temporarily inject
    global CONFIGS
    saved = dict(CONFIGS)
    CONFIGS.update(ladder_configs)
    results["link_class_ladder_n2"] = eval_set(["L1", "L2", "L3", "L4"])
    # mixed
    mixed = {"M1": (2, 900.0, 1.0), "M2": (2, 100.0, 5.0),
             "M3": (4, 50.0, 5.0), "M4": (4, 25.0, 10.0)}
    CONFIGS.update(mixed)
    results["mixed"] = eval_set(["M1", "M2", "M3", "M4"])
    # link-class separated: drop near-degenerate 4-dev NVLink (C), keep A,B,D
    results["drop_C_ABC?_ABD"] = eval_set(["A", "B", "D"])
    # link-class separated: A (2dev NVLink), B (2dev PCIe), D (4dev 25GbE) already above
    # try A,C,D (drop B)
    results["drop_B_ACD"] = eval_set(["A", "C", "D"])
    CONFIGS.clear()
    CONFIGS.update(saved)

    # choose admitted: first admitted set
    admitted_name = None
    for pref in ("mixed", "drop_B_ACD", "link_class_ladder_n2", "drop_C_ABC?_ABD"):
        if pref in results and results[pref]["admitted"]:
            admitted_name = pref
            break
    if admitted_name is None:
        for name, res in results.items():
            if res["admitted"]:
                admitted_name = name
                break

    out = {"size_bytes": SIZE_BYTES, "count": COUNT,
           "corrected_model": "gpu-sim-8sy (commit 70a40cf8)",
           "candidate_sets": results,
           "admitted_set": admitted_name,
           "rejected": {k: v for k, v in results.items() if k != admitted_name}}
    with open(os.path.join(os.path.dirname(__file__), "admissibility.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({"admitted_set": admitted_name}, indent=2))
    for name, res in results.items():
        print(name, "ADMITTED" if res["admitted"] else "REJECTED", res["order"])
        for p in res["pairs"]:
            print("   ", p["pair"], "d_pred=%.1fus" % p["d_pred_us"], "rel=%.1f%%" % (p["rel_margin"]*100), "bw_ratio=%.2f" % p["bw_ratio"], "admissible=%s" % p["admissible"])


if __name__ == "__main__":
    main()
