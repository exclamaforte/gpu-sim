"""gpu-sim-vvty step 1: activation feasibility of every published pp>1 row at full storage.

Re-scores each published pp>1 winner config through the PRODUCTION score path
(epic_cluster/parallelism_search.py) with the published hw (recompute=0.0) and
records activation GB/stage vs the 80 GB budget. No reimplementation: the hw
dict is built exactly as scripts/scaleout_answers.py answer() builds it.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from scripts.scaleout_answers import build_spec, MODELS, PEAK, HBM_GB
from epic_cluster import parallelism_search as ps

_HERE = os.path.dirname(os.path.abspath(__file__))
ANS = json.load(open(os.path.join(_HERE, "..", "..", "experiments/gpu-sim-scaleout/answers.json")))


def parse_batch(key):
    if "batch=" in key:
        return int(key.split("batch=")[1])
    return 1


def parse_conv(key):
    return key.split(", batch=")[0]


def main():
    rows = []
    for fab, blob in ANS.items():
        nodes = int(fab.split("x")[0])
        spec = build_spec(nodes)
        for model_name, cells in blob["models"].items():
            model = MODELS[model_name]
            for key, r in cells.items():
                if r is None or r.get("pp", 0) <= 1:
                    continue
                batch = parse_batch(key)
                conv = parse_conv(key)
                scale = {"ring all_to_all": 1.0, "default all_gather": 1.9922}[conv]
                hw = {"peak_flops": PEAK, "mem_gb": HBM_GB, "spec": spec,
                      "term_scale": {"cp": scale}, "batch": batch, "recompute": 0.0}
                c = ps.score(r["tp"], r["pp"], r["cp"], r["dp"], r["zero"], hw, model)
                rows.append({
                    "fabric": fab, "model": model_name, "convention": conv, "batch": batch,
                    "config": r["config"], "pp": r["pp"],
                    "act_gb": round(c.mem_act_gb, 2),
                    "attn_ws_gb": round(c.mem_attn_ws_gb, 2),
                    "total_mem_gb": round(c.memory_gb, 1),
                    "budget_gb": HBM_GB,
                    "act_headroom_gb": round(HBM_GB - c.mem_act_gb, 2),
                })
    rows.sort(key=lambda d: d["act_headroom_gb"])
    out = {"rows": rows, "n_pp_gt_1": len(rows),
           "tightest": rows[0] if rows else None,
           "all_act_fit": all(d["act_headroom_gb"] > 0 for d in rows)}
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "act_feasibility.json")
    json.dump(out, open(path, "w"), indent=1)
    print(f"{len(rows)} pp>1 rows; tightest act headroom: "
          f"{rows[0]['act_headroom_gb']} GB ({rows[0]['fabric']} {rows[0]['model']} "
          f"{rows[0]['convention']} batch={rows[0]['batch']} {rows[0]['config']}, "
          f"act {rows[0]['act_gb']} GB)")
    print("all act fit in budget:", out["all_act_fit"])


if __name__ == "__main__":
    main()
