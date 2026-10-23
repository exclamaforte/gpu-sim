#!/usr/bin/env python3
import json, math, pathlib, sys, numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

HERE = pathlib.Path(__file__).resolve().parent
DATASET = pathlib.Path(__file__).resolve().parents[2] / "experiments" / "gpu-sim-2al" / "dataset.jsonl"

def phys(r):
    n=r["size"]; b=r["block"]
    bm=2*n*4.0
    fl=n*(64.0 if r["family"]=="matmul" else 1.0)
    ai=fl/bm
    grid=min((n+b-1)//b,65535)
    waves=grid/170.0
    return [math.log1p(n), b, bm, fl, ai, grid, waves, math.log2(n)]

def size_only(r):
    return [math.log1p(r["size"])]

def split(rows):
    train=[]; test=[]
    for fam in sorted(set(r["family"] for r in rows)):
        fr=sorted([r for r in rows if r["family"]==fam], key=lambda r:(r["size"],r["block"]))
        cut=int(0.8*len(fr))
        train.extend(fr[:cut]); test.extend(fr[cut:])
    return train,test

def train_gbt(Xtr,ytr,Xte):
    g=HistGradientBoostingRegressor(max_iter=300,max_depth=6,learning_rate=0.05,min_samples_leaf=5,random_state=0)
    g.fit(Xtr,ytr)
    return g.predict(Xte)

def med_rel(preds, trues):
    rels=[abs(math.exp(p)-math.exp(t))/math.exp(t) for p,t in zip(preds,trues)]
    return float(np.median(rels))

def main():
    rows=[json.loads(l) for l in DATASET.read_text().splitlines()]
    train,test=split(rows)
    Xtr_raw=np.array([phys(r)+r["num33"] for r in train],dtype=float)
    Xte_raw=np.array([phys(r)+r["num33"] for r in test],dtype=float)
    mu=Xtr_raw.mean(0); sd=Xtr_raw.std(0); sd[sd==0]=1
    Xtr=(Xtr_raw-mu)/sd; Xte=(Xte_raw-mu)/sd
    ytr=np.array([r["y"] for r in train]); yte=np.array([r["y"] for r in test])
    pred_full=train_gbt(Xtr,ytr,Xte)
    Xtr0=np.array([size_only(r) for r in train],dtype=float); Xte0=np.array([size_only(r) for r in test],dtype=float)
    mu0=Xtr0.mean(0); sd0=Xtr0.std(0); sd0[sd0==0]=1
    Xtr0=(Xtr0-mu0)/sd0; Xte0=(Xte0-mu0)/sd0
    pred0=train_gbt(Xtr0,ytr,Xte0)
    gate_by_family={}
    baseline={"elementwise":0.506,"matmul":0.260,"reduction":0.462,"softmax":0.415,"transpose":0.245}
    for fam in sorted(set(r["family"] for r in test)):
        idx=[i for i,r in enumerate(test) if r["family"]==fam]
        p=[pred_full[i] for i in idx]; t=[yte[i] for i in idx]; p0=[pred0[i] for i in idx]
        gate_by_family[fam]={
            "n": len(idx),
            "gbt_median_rel_error": med_rel(p,t),
            "gbt_log_mae": float(np.mean(np.abs(np.array(p)-np.array(t)))),
            "sizeonly_median_rel_error": med_rel(p0,t),
        }
    pooled_med=med_rel(pred_full,yte)
    passes=[f for f in gate_by_family if gate_by_family[f]["gbt_median_rel_error"]<=0.25]
    if len(passes)==len(gate_by_family):
        verdict="PASS-ALL"
    elif len(passes)>=1:
        verdict="PARTIAL"
    else:
        verdict="GAP-REMAINS"
    X=np.array([phys(r)+r["num33"] for r in rows])
    informative=[i for i in range(X.shape[1]) if len(set(X[:,i]))>1]
    out={
        "gate_statistic_by_family": gate_by_family,
        "baseline_from_nk0": baseline,
        "pooled_median_rel_error": pooled_med,
        "verdict": verdict,
        "informative_slots": len(informative),
        "feature_spec": "phys(8) + num33(33) = 41 slots",
    }
    (HERE/"bwn_results.json").write_text(json.dumps(out,indent=2)+"\n")
    with open(HERE/"predictions_bwn.jsonl","w") as f:
        for r,p in zip(test,pred_full):
            f.write(json.dumps({"family":r["family"],"size":r["size"],"block":r["block"],"y_true":r["y"],"y_pred":float(p)})+"\n")
    with open(HERE/"predictions_sizeonly.jsonl","w") as f:
        for r,p in zip(test,pred0):
            f.write(json.dumps({"family":r["family"],"size":r["size"],"block":r["block"],"y_true":r["y"],"y_pred":float(p)})+"\n")
    print(json.dumps(out,indent=2))

if __name__=="__main__":
    main()
