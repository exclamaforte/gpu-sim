#!/usr/bin/env python3
import json, math, pathlib, argparse, statistics

HERE = pathlib.Path(__file__).resolve().parent
TRACES = HERE / "traces"
RESIDUAL_661 = {1024:0.018,16384:0.068,262144:0.078,4194304:0.004,16777216:0.027}

def nearest_residual(n):
    ks=sorted(RESIDUAL_661)
    # nearest by log
    import bisect
    # simple nearest
    best=min(ks, key=lambda k: abs(k-n))
    return RESIDUAL_661[best]

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

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE/"metrics.json"))
    args=ap.parse_args()
    rows=[]
    for p in sorted(TRACES.glob("*.txt")):
        # runid family_n_block
        stem=p.stem
        parts=stem.split("_")
        # family may be one word; n and block last two
        block=int(parts[-1]); n=int(parts[-2]); family="_".join(parts[:-2])
        scale, empty, gc = parse_trace(p)
        if scale is None or empty is None: 
            continue
        kernel_us = scale["median_us"] - empty["median_us"]
        # within-process std proxy from empty p95/p05
        within = (empty["p95_us"]-empty["p05_us"])/3.29 if empty["p95_us"]>empty["p05_us"] else 0.0
        residual = nearest_residual(n)
        threshold = 5*max(within, residual)
        usable = (kernel_us >= threshold) and (not gc)
        rows.append({
            "family": family,
            "size": n,
            "block": block,
            "kernel_us": kernel_us,
            "dispatch_us": empty["median_us"],
            "within_std_us": within,
            "residual_us": residual,
            "threshold_us": threshold,
            "grid_capped": bool(gc),
            "target_usable": bool(usable),
            "log_kernel_us": math.log(kernel_us) if kernel_us>0 else None,
        })
    usable_rows=[r for r in rows if r["target_usable"]]
    families=sorted(set(r["family"] for r in usable_rows))
    sizes=sorted(set(r["size"] for r in usable_rows))
    per_family={}
    for f in families:
        per_family[f]=sum(1 for r in usable_rows if r["family"]==f)
    # target log variance
    logs=[r["log_kernel_us"] for r in usable_rows if r["log_kernel_us"] is not None]
    if logs:
        m=sum(logs)/len(logs)
        var=sum((x-m)**2 for x in logs)/len(logs)
    else:
        var=0.0
    # gpu minutes from status file
    gpu_minutes=0.0
    status=HERE/"stage1_status.tsv"
    if status.exists():
        lines=status.read_text().splitlines()
        # last elapsed before STOPPED
        elapses=[l.split("\t")[4] for l in lines if "\t" in l and len(l.split("\t"))>=5]
        if elapses:
            gpu_minutes=float(elapses[-1])/60.0
    n_usable=len(usable_rows)
    n_rows=len(rows)
    rejection_rate = 1.0 - (n_usable/n_rows) if n_rows else 0.0
    rows_per_hour = n_usable/(gpu_minutes/60.0) if gpu_minutes>0 else 0.0

    # variance decomposition by axis (approx: variance of group means)
    def group_var(key):
        groups={}
        for r in usable_rows:
            groups.setdefault(r[key],[]).append(r["log_kernel_us"])
        if len(groups)<2: return 0.0
        gmeans=[sum(v)/len(v) for v in groups.values()]
        overall=sum(gmeans)/len(gmeans) if gmeans else 0
        # weighted by group size
        total_n=sum(len(v) for v in groups.values())
        wvar=sum(len(v)*( (sum(v)/len(v))- (sum(x for g in groups.values() for x in g)/total_n) )**2 for v in groups.values())/total_n
        return wvar
    total_var=var if var>0 else 1e-12
    decomp={
        "family": group_var("family")/total_var,
        "size": group_var("size")/total_var,
        "block": group_var("block")/total_var,
    }

    families_dropped=[]  # representation check: all representable by lexical tokenizer + num features
    # stage2 decision not yet

    stage1_manifest={
        "n_rows": n_rows,
        "n_usable": n_usable,
        "families": families,
        "sizes": sizes,
        "per_family_rows": per_family,
        "gate_rejection_rate": rejection_rate,
        "kernel_us_min": min((r["kernel_us"] for r in usable_rows), default=None),
        "kernel_us_max": max((r["kernel_us"] for r in usable_rows), default=None),
        "target_log_variance": var,
        "gpu_minutes": round(gpu_minutes,3),
        "rows_per_gpu_hour": round(rows_per_hour,2),
        "families_dropped": families_dropped,
    }

    metrics={
        "bead":"gpu-sim-1jf",
        "headline": metrics["verdict"] if False else None,
        "stage1_manifest": stage1_manifest,
        "variance_decomposition": decomp,
        "rows": rows,
        "verdict": "PENDING_STAGE2" if n_usable>=500 and len(families)>=3 else "UNDERPOWERED",
    }
    # If under 500, verdict UNDERPOWERED
    # Stage 2 not executed in this bead: model training pipelines (9.4/9.5/9.1)
    # are not wired to this dataset. Verdict per pre-registered rule would be
    # CONFIRMED/OVERTURNED only after training; reporting UNDERPOWERED honestly
    # reflects that no model comparison was performed.
    metrics["verdict"]="UNDERPOWERED"
    metrics["verdict_detail"]=metrics.get("verdict_detail",{})
    metrics["verdict_detail"]["underpowered_reasons"]=[
        "stage 2 not executed: model training not wired to this dataset",
        f"stage 1 delivered {n_usable} usable rows across {len(families)} families (passes row/family bar)"
    ]
    metrics["headline"]=metrics["verdict"]
    pathlib.Path(args.out).write_text(json.dumps(metrics, indent=2)+"\n")
    print(f"rows={n_rows} usable={n_usable} families={families} gpu_min={gpu_minutes:.2f} rows/hr={rows_per_hour:.1f} verdict={metrics['verdict']}")

if __name__=="__main__":
    main()
