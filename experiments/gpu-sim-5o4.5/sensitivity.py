"""5o4.5 sensitivity study (CPU only)."""
import sys, os, json, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from epic_cluster import parallelism_search as ps

N = 8
BASE = {"peak_flops": 989e12, "mem_gb": 80.0, "fabric": "nvlink4_h100",
        "alpha_s": 1.5e-6, "bw_gbps": 450.0}

def top1(N, hw, rank_by="sum"):
    _, feas = ps.search(N, hw, rank_by=rank_by)
    return feas[0] if feas else None

def key(c):
    return (c.tp, c.pp, c.cp, c.dp, c.zero)

# baseline
base_winner = top1(N, dict(BASE))
print("baseline winner:", key(base_winner), base_winner.step_sum*1000)

# 1. one-at-a-time break-even for alpha and bw
break_even = {}
for param, rng in [("alpha_s", [0.5e-6, 1e-6, 1.5e-6, 2e-6, 3e-6, 5e-6]),
                   ("bw_gbps", [200, 300, 350, 400, 450, 500, 600])]:
    winners = []
    for v in rng:
        hw = dict(BASE); hw[param] = v
        w = top1(N, hw)
        winners.append((v, key(w), w.step_sum*1000))
    break_even[param] = winners
    print(param, winners)

# peak flops
for pf in [500e12, 700e12, 989e12, 1200e12]:
    hw = dict(BASE); hw["peak_flops"] = pf
    w = top1(N, hw)
    print("peak_flops", pf, key(w), w.step_sum*1000)

# overlap policy
for rb in ["sum", "max"]:
    w = top1(N, dict(BASE), rank_by=rb)
    print("rank_by", rb, key(w), getattr(w, "step_sum" if rb=="sum" else "step_max")*1000)

# 2. joint band
alphas = [0.75e-6, 1.0e-6, 1.5e-6, 2.0e-6, 3.0e-6]
bws = [315, 360, 405, 450, 540]
winners_dist = collections_counter = {}
from collections import Counter
dist = Counter()
inversions = []
base_top10 = [key(c) for c in ps.search(N, dict(BASE))[1][:10]]
for a in alphas:
    for b in bws:
        for rb in ["sum","max"]:
            hw = dict(BASE); hw["alpha_s"]=a; hw["bw_gbps"]=b
            feas = ps.search(N, hw, rank_by=rb)[1]
            if feas:
                dist[key(feas[0])] += 1
                top10 = [key(c) for c in feas[:10]]
                if top10 != base_top10:
                    inversions.append({"alpha":a,"bw":b,"rank_by":rb,"winner":list(top10[0]),"base_winner":list(base_top10[0])})

# stability: how many distinct winners
print("winner distribution:", dict(dist))
print("num distinct winners:", len(dist))
print("num inversions from base top10:", len(inversions))

metrics = {
  "baseline_winner": list(key(base_winner)),
  "baseline_step_ms": base_winner.step_sum*1000,
  "break_even": {k: [(v, list(kk), ms) for v,kk,ms in vals] for k,vals in break_even.items()},
  "winner_distribution": {str(k):v for k,v in dist.items()},
  "num_distinct_winners": len(dist),
  "num_inversions": len(inversions),
  "inversions_sample": inversions[:10],
  "verdict": "not separable at this margin" if len(dist)>1 else "robust"
}
with open("experiments/gpu-sim-5o4.5/metrics.json","w") as f:
    json.dump(metrics, f, indent=2)
print("written")
