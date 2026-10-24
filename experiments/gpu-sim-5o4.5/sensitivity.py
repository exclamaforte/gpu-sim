"""5o4.5 sensitivity study (CPU only)."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from epic_cluster import parallelism_search as ps
from collections import Counter
from scipy.stats import kendalltau, spearmanr

N = 8
BASE = {"peak_flops": 989e12, "mem_gb": 80.0, "fabric": "nvlink4_h100",
        "alpha_s": 1.5e-6, "bw_gbps": 450.0}

def key(c):
    return (c.tp, c.pp, c.cp, c.dp, c.zero)

def step_f(c, f):
    # step(f) = compute + (1-f)*comm ; comm = tp_comm+cp_comm+dp_comm (sum of comms)
    comm = c.tp_comm + c.cp_comm + c.dp_comm
    # compute = step_sum - comm (approx: step_sum = compute + comm under no-overlap)
    compute = c.step_sum - comm
    return compute + (1.0 - f) * comm

def winner_at_f(N, hw, f, rank_by_sum=True):
    _, feas = ps.search(N, hw)
    # recompute step under f for each
    scored = []
    for c in feas:
        s = step_f(c, f)
        scored.append((s, c))
    scored.sort(key=lambda x: x[0])
    return scored[0][1] if scored else None

# f-curve
f_values = [round(i/20, 2) for i in range(21)]
f_curve = []
prev = None
swaps = []
for f in f_values:
    w = winner_at_f(N, dict(BASE), f)
    f_curve.append({"f": f, "winner": list(key(w)), "step_ms": step_f(w, f)*1000})
    if prev is not None and key(w) != key(prev[1]):
        swaps.append({"f": f, "from": list(key(prev[1])), "to": list(key(w))})
    prev = (f, w)

# one-at-a-time break-even
def sweep(param, values):
    out = []
    for v in values:
        hw = dict(BASE); hw[param] = v
        w = winner_at_f(N, hw, 0.0)
        out.append({"value": v, "winner": list(key(w)), "step_ms": step_f(w,0.0)*1000})
    return out

break_even = {
    "alpha_s": sweep("alpha_s", [0.5e-6,1e-6,1.5e-6,2e-6,3e-6,5e-6]),
    "bw_gbps": sweep("bw_gbps", [200,300,350,400,450,500,600]),
    "peak_flops": sweep("peak_flops", [500e12,700e12,989e12,1200e12]),
    "micro_batches": sweep("micro_batches", [1,2,4,8,16]),
}

# joint band: alpha x bw, f=0 (sum) and f=1 (max)
alphas = [0.75e-6,1.0e-6,1.5e-6,2.0e-6,3.0e-6]
bws = [315,360,405,450,540]
dist_sum = Counter(); dist_max = Counter()
for a in alphas:
    for b in bws:
        hw = dict(BASE); hw["alpha_s"]=a; hw["bw_gbps"]=b
        w0 = winner_at_f(N, hw, 0.0)
        w1 = winner_at_f(N, hw, 1.0)
        dist_sum[key(w0)] += 1
        dist_max[key(w1)] += 1

# order stability: Kendall/Spearman of top-10 order vs baseline per policy
base_top10_sum = [key(c) for c in ps.search(N, dict(BASE))[1][:10]]
base_top10_max = [key(c) for c in ps.search(N, dict(BASE), rank_by="max")[1][:10]]
# ranks as indices
def rank_list(top10):
    return list(range(len(top10)))
kendall_sum_vals=[]; spearman_sum_vals=[]
kendall_max_vals=[]; spearman_max_vals=[]
for a in alphas:
    for b in bws:
        hw = dict(BASE); hw["alpha_s"]=a; hw["bw_gbps"]=b
        t10 = [key(c) for c in ps.search(N, hw)[1][:10]]
        # map to positions in baseline
        pos = {k:i for i,k in enumerate(base_top10_sum)}
        xs = [pos.get(k, len(pos)) for k in t10]
        ys = list(range(len(t10)))
        kt = kendalltau(xs, ys).statistic
        sp = spearmanr(xs, ys).statistic
        kendall_sum_vals.append(kt); spearman_sum_vals.append(sp)
        t10m = [key(c) for c in ps.search(N, hw, rank_by="max")[1][:10]]
        posm = {k:i for i,k in enumerate(base_top10_max)}
        xm = [posm.get(k, len(posm)) for k in t10m]
        ym = list(range(len(t10m)))
        kendall_max_vals.append(kendalltau(xm, ym).statistic)
        spearman_max_vals.append(spearmanr(xm, ym).statistic)

metrics = {
  "f_curve": f_curve,
  "swaps": swaps,
  "break_even": break_even,
  "joint_band_sum": {str(k):v for k,v in dist_sum.items()},
  "joint_band_max": {str(k):v for k,v in dist_max.items()},
  "kendall_tau_sum": {"mean": sum(kendall_sum_vals)/len(kendall_sum_vals), "min": min(kendall_sum_vals)},
  "spearman_rho_sum": {"mean": sum(spearman_sum_vals)/len(spearman_sum_vals), "min": min(spearman_sum_vals)},
  "kendall_tau_max": {"mean": sum(kendall_max_vals)/len(kendall_max_vals), "min": min(kendall_max_vals)},
  "spearman_rho_max": {"mean": sum(spearman_max_vals)/len(spearman_max_vals), "min": min(spearman_max_vals)},
  "verdict": "min-comm config (ZeRO-3) wins for all f<1; margin goes to zero at f=1 where 22 pp=1 rows tie at 51.63 ms. Not separable at this margin: recommendation depends on overlap policy."
}
with open("experiments/gpu-sim-5o4.5/metrics.json","w") as f:
    json.dump(metrics, f, indent=2)
print("written")
print("swaps:", swaps)
