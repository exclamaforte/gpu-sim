#!/usr/bin/env python3
"""MEASURE phase (GPU, queued via ts): run prebuilt binaries, compute
kernel_us = scale_median - empty_median, apply bk9 gate, append rows."""
import argparse, json, statistics, time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from epic_u2o.safe_harness import SafeHarness
from scripts.grid_limits import grid_capped as _grid_capped

RESIDUAL = {1024:0.018,16384:0.068,262144:0.078,4194304:0.004,16777216:0.027}
def nearest_residual(n):
    ks=sorted(RESIDUAL); return RESIDUAL[min(ks,key=lambda k:abs(k-n))]

def med(xs): return statistics.median(xs) if xs else 0.0

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--build-dir", default=None)
    ap.add_argument("--out", default=None)
    args=ap.parse_args()
    here=Path(__file__).resolve().parent
    build=Path(args.build_dir) if args.build_dir else (here/"build")
    out=Path(args.out) if args.out else (here/"measured.jsonl")
    manifest=json.loads((build/"manifest.json").read_text())
    specs=manifest["specs"]
    h=SafeHarness(timeout_sec=120)
    # load already measured keys
    done=set()
    if out.exists():
        for line in out.read_text().splitlines():
            if not line.strip(): continue
            try:
                r=json.loads(line); done.add((r["family"], r["build_idx"]))
            except: pass
    t0=time.time()
    n_ok=0
    with open(out,"a") as fh:
        for d in specs:
            key=(d["family"], d["build_idx"])
            if key in done:
                continue
            exe=Path(d["exe"])
            if not exe.exists():
                continue
            # empty baseline
            g=tuple(d["grid"]); b=tuple(d["block"])
            empty_key="empty_"+"_".join(map(str,list(g)+list(b)))
            empty_exe=build/"bin"/(empty_key+".out")
            empty_trials=[]
            if empty_exe.exists():
                res=h.run_kernel(str(empty_exe))
                if res.success:
                    for line in res.stdout.splitlines():
                        line=line.strip()
                        if line.startswith("{"):
                            try:
                                o=json.loads(line)
                                if "time_ms" in o: empty_trials.append(o["time_ms"])
                            except: pass
            # scale run
            res=h.run_kernel(str(exe))
            if not res.success:
                row={"family":d["family"],"build_idx":d["build_idx"],"error":f"run_fail rc={res.returncode}","target_usable":False}
                fh.write(json.dumps(row)+"\n"); fh.flush()
                continue
            scale_trials=[]
            for line in res.stdout.splitlines():
                line=line.strip()
                if line.startswith("{"):
                    try:
                        o=json.loads(line)
                        if "time_ms" in o: scale_trials.append(o["time_ms"])
                    except: pass
            if len(scale_trials)<10:
                row={"family":d["family"],"build_idx":d["build_idx"],"error":f"few_trials {len(scale_trials)}","target_usable":False}
                fh.write(json.dumps(row)+"\n"); fh.flush()
                continue
            scale_med_ms=med(scale_trials)
            empty_med_ms=med(empty_trials) if empty_trials else 0.0
            kernel_us=(scale_med_ms-empty_med_ms)*1000.0
            # within proxy from empty baseline p95/p05 (1jf style)
            if len(empty_trials)>=5:
                es=sorted(empty_trials)
                p05=es[max(0,int(len(es)*0.05))]
                p95=es[min(len(es)-1,int(len(es)*0.95))]
                within_us=(p95-p05)/3.29*1000.0
            else:
                within_us=0.0
            residual=nearest_residual(d["problem_size"])
            threshold=5*max(within_us, residual) if (within_us>0 or residual>0) else 0.0
            md = d.get("metadata", {})
            compute_bound = bool(md.get("compute_bound", False))
            # grid capped: use the ACTUAL grid from the spec, not a size/block proxy
            grid = d["grid"]
            gc = _grid_capped(grid)
            # family-aware threshold: flop-bound (compute-bound) kernels use the
            # irreducible residual as the noise floor, not the empty-baseline
            # variance (which is inflated by clock boosting on trivial kernels).
            md = d.get("metadata", {})
            compute_bound = bool(md.get("compute_bound", False))
            if compute_bound:
                threshold = 5 * residual if residual > 0 else threshold
            usable = (kernel_us >= threshold) and (not gc) and (kernel_us>0)
            rejection_reason = None
            if not usable:
                if kernel_us <= 0:
                    rejection_reason = "nonpositive_kernel_us"
                elif gc:
                    rejection_reason = "grid_capped"
                elif kernel_us < threshold:
                    rejection_reason = "below_threshold"
                else:
                    rejection_reason = "unknown"
            import math
            row={
                "family":d["family"],
                "build_idx":d["build_idx"],
                "op_type":d["op_type"],
                "size":d["problem_size"],
                "block":d["block"][0]*d["block"][1]*d["block"][2] if len(d["block"])>=3 else d["block"][0],
                "grid":d["grid"],
                "kernel_us":kernel_us,
                "dispatch_us":empty_med_ms*1000.0,
                "empty_p05_ms": (sorted(empty_trials)[max(0,int(len(empty_trials)*0.05))] if empty_trials else None),
                "empty_p95_ms": (sorted(empty_trials)[min(len(empty_trials)-1,int(len(empty_trials)*0.95))] if empty_trials else None),
                "within_std_us":within_us,
                "residual_us":residual,
                "threshold_us":threshold,
                "grid_capped":bool(gc),
                "target_usable":bool(usable),
                "rejection_reason":rejection_reason,
                "log_kernel_us": math.log(kernel_us) if kernel_us>0 else None,
                "phys":d.get("phys",{}),
                "n_trials":len(scale_trials),
            }
            fh.write(json.dumps(row)+"\n"); fh.flush()
            done.add(key)
            n_ok+=1
            if n_ok%25==0:
                print(f"measured {n_ok} rows, elapsed {time.time()-t0:.0f}s", flush=True)
    print(f"DONE measured {n_ok} new rows -> {out}")

if __name__=="__main__":
    main()
