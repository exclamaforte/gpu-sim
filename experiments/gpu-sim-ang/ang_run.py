#!/usr/bin/env python3
import json, math, statistics, pathlib, sys
REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO/"experiments/gpu-sim-u2o.17.3"))
import sim_trace_173 as S

FIX = REPO / "experiments/gpu-sim-u2o.17.5/canonical_fixture.json"
KNEE = REPO / "experiments/gpu-sim-u2o.17.3/knee_cases.json"

def build_points():
    fix = json.loads(FIX.read_text())
    cases = list(fix["cases"])
    sup = json.loads(KNEE.read_text())
    cases += sup["cases"]
    med = {}
    for c in cases:
        med.setdefault((c["op_class"], str(c["shapes"])), []).append((c["device_ms_median"], c))
    per_cls = {}
    for (cls, sh), lst in med.items():
        vs = sorted(v for v, _ in lst)
        per_cls.setdefault(cls, []).append((S.case_size if False else None, sh, vs[len(vs)//2], lst[0][1]))
    # rebuild like S
    per_cls = {}
    for (cls, sh), lst in med.items():
        vs = sorted(v for v, _ in lst)
        per_cls.setdefault(cls, []).append((None, sh, vs[len(vs)//2], lst[0][1]))
    # compute size via S.case_size-like: use size_of
    result = {}
    for cls, pts in per_cls.items():
        out=[]
        for _, sh, med_ms, rep in pts:
            s = json.loads(sh)
            if cls == "matmul":
                sz = 2 * s[0]**3
            else:
                p=1
                for d in s: p*=d
                sz=p
            out.append((sz, sh, med_ms, rep))
        out.sort()
        result[cls]=out
    return result

def loglog_fit(pts):
    xs=[math.log(x) for x,_ in pts]; ys=[math.log(y) for _,y in pts]
    n=len(xs); mx=sum(xs)/n; my=sum(ys)/n
    den=sum((x-mx)**2 for x in xs)
    b=sum((x-mx)*(y-my) for x,y in zip(xs,ys))/den if den else 0.0
    return my-b*mx, b

def interp_predict(train, sz):
    # interior interpolation in log-log between nearest neighbors
    train=sorted(train)
    if sz <= train[0][0]:
        a,b=loglog_fit(train[:3]); return math.exp(a+b*math.log(sz))
    if sz >= train[-1][0]:
        a,b=loglog_fit(train[-3:]); return math.exp(a+b*math.log(sz))
    # find bracket
    for i in range(len(train)-1):
        if train[i][0] <= sz <= train[i+1][0]:
            (x0,y0),(x1,y1)=train[i],train[i+1]
            t=(math.log(sz)-math.log(x0))/(math.log(x1)-math.log(x0))
            return math.exp(math.log(y0)+t*(math.log(y1)-math.log(y0)))
    a,b=loglog_fit(train); return math.exp(a+b*math.log(sz))

def regime_predict(train, sz, regimes=2):
    # two-regime: split train by median size
    med_sz=sorted(x for x,_ in train)[len(train)//2]
    subset=[(x,y) for x,y in train if (x<=med_sz)==(sz<=med_sz)]
    if len(subset)<3:
        subset=train
    a,b=loglog_fit(subset)
    return math.exp(a+b*math.log(sz))

def run_config(points_by_class, use_l1, use_l2):
    out={}
    for cls, points in points_by_class.items():
        if len(points)<3:
            continue
        errs=[]
        for i,(sz,sh,med_ms,rep) in enumerate(points):
            train=[(p[0],p[2]) for j,p in enumerate(points) if j!=i]
            if use_l1:
                pred=interp_predict(train, sz)
            elif use_l2 and cls=="elementwise":
                pred=regime_predict(train, sz)
            else:
                a,b=loglog_fit(train); pred=math.exp(a+b*math.log(sz))
            # both: apply l2 for elementwise, l1 otherwise? For both combine: l1 interpolation + regime
            if use_l1 and use_l2 and cls=="elementwise":
                # regime selection then interpolate within regime
                med_sz=sorted(x for x,_ in train)[len(train)//2]
                subset=[(x,y) for x,y in train if (x<=med_sz)==(sz<=med_sz)]
                if len(subset)>=2:
                    pred=interp_predict(subset, sz)
                else:
                    pred=interp_predict(train, sz)
            ef=(pred-med_ms)/med_ms
            errs.append(abs(ef))
        out[cls]=float(__import__('statistics').median(errs))
    return out

def main():
    pts=build_points()
    # only elementwise, matmul, reduce gate classes
    gate_classes=["elementwise","matmul","reduce"]
    pts={k:v for k,v in pts.items() if k in gate_classes}
    configs={
        "baseline": (False,False),
        "+l1": (True,False),
        "+l2": (False,True),
        "+both": (True,True),
    }
    results={}
    for name,(l1,l2) in configs.items():
        r=run_config(pts,l1,l2)
        results[name]=r
        print(name, r)
    high_spread=[{"op_class":c["op_class"],"shapes":c["shapes"],"spread_pct":round(c["device_ms_spread_pct_of_median"],2)} for c in json.loads((REPO/"experiments/gpu-sim-u2o.17.5/canonical_fixture.json").read_text())["cases"] if c.get("device_ms_spread_pct_of_median",0)>25]
    metrics={
        "bead":"gpu-sim-ang",
        "step0_baseline": results["baseline"],
        "step0_pinned": {"elementwise":0.969,"matmul":0.409,"reduce":0.473},
        "levers": results,
        "verdict": "GAP-REMAINS",
        "high_spread_cases": high_spread,
        "bar": 0.25,
        "bar_criterion": "u2o.17: median |sim-measured|/measured <= 25% per op class",
    }
    # verdict: if any config all <=0.25 -> PASS-ALL etc
    best=None
    for name,r in results.items():
        if all(v<=0.25 for v in r.values()):
            best=name; break
    if best:
        metrics["verdict"]="PASS-ALL"
    else:
        # partial?
        passes=max(sum(1 for v in r.values() if v<=0.25) for r in results.values())
        metrics["verdict"]="PARTIAL" if passes>=2 else "GAP-REMAINS"
    pathlib.Path("experiments/gpu-sim-ang/metrics.json").write_text(json.dumps(metrics,indent=2)+"\n")
    print("verdict", metrics["verdict"])

if __name__=="__main__":
    main()
