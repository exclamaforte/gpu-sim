#!/usr/bin/env python3
import json, math, pathlib, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.grid_limits import grid_capped as _grid_capped
from scripts import grid_limits
GRID_CAPPED = _grid_capped

def _row_grid_capped(grid):
    return grid_limits.grid_capped(grid)
TR="experiments/gpu-sim-8zqr/traces_fk"
RESIDUAL={1024:0.018,16384:0.068,262144:0.078,4194304:0.004,16777216:0.027}
def nearest_residual(n):
    ks=sorted(RESIDUAL); return RESIDUAL[min(ks,key=lambda k:abs(k-n))]
def parse_trace(path):
    scale=None; empty=None; gc=False
    for line in path.read_text(errors="replace").splitlines():
        line=line.strip()
        if not line.startswith("{"): continue
        try: rec=json.loads(line)
        except: continue
        if rec.get("summary"):
            if rec["mode"]=="scale": scale=rec
            elif rec["mode"]=="empty": empty=rec
            gc = gc or bool(rec.get("grid_capped"))
    return scale, empty, gc
rows=[]
for p in sorted(pathlib.Path(TR).glob("*.txt")):
    parts=p.stem.split("_")
    block=int(parts[-1]); n=int(parts[-2]); family="_".join(parts[:-2])
    scale,empty,gc=parse_trace(p)
    if scale is None or empty is None: continue
    kernel_us=scale["median_us"]-empty["median_us"]
    within=(empty["p95_us"]-empty["p05_us"])/3.29 if empty["p95_us"]>empty["p05_us"] else 0.0
    residual=nearest_residual(n)
    threshold=5*max(within,residual)
    compute_bound = (family == "gemm_naive")
    if compute_bound:
        threshold = 5 * residual if residual > 0 else threshold
    grid_dim = (n + block - 1) // block if block > 0 else 0
    gc = bool(gc) or _row_grid_capped([grid_dim, 1, 1])
    usable=(kernel_us>=threshold) and (not gc) and (kernel_us>0)
    rejection_reason = None
    if not usable:
        if kernel_us <= 0: rejection_reason="nonpositive_kernel_us"
        elif gc: rejection_reason="grid_capped"
        elif kernel_us < threshold: rejection_reason="below_threshold"
        else: rejection_reason="unknown"
    bm=2*n*4.0
    flops=n  # memory-access family
    ai=flops/bm
    grid_dim=(n+block-1)//block
    waves=grid_dim/170.0
    rows.append({
        "family":family,"size":n,"block":block,
        "kernel_us":kernel_us,"dispatch_us":empty["median_us"],
        "within_std_us":within,"residual_us":residual,"threshold_us":threshold,
        "grid_capped":bool(gc),"target_usable":bool(usable),"rejection_reason":rejection_reason,
        "y":math.log(kernel_us) if kernel_us>0 else None,
        "phys":{"n":n,"block":block,"bm":bm,"flops":flops,"ai":ai,"grid":grid_dim,"waves":waves},
    })
# write usable-only and all
out_all=pathlib.Path("experiments/gpu-sim-8zqr/powered_rows.jsonl")
out_all.write_text("\n".join(json.dumps(r) for r in rows)+"\n")
usable=[r for r in rows if r["target_usable"]]
out_u=pathlib.Path("experiments/gpu-sim-8zqr/powered_usable.jsonl")
out_u.write_text("\n".join(json.dumps(r) for r in usable)+"\n")
print("total",len(rows),"usable",len(usable))
from collections import Counter
print(Counter(r["family"] for r in usable))
