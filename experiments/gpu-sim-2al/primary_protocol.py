#!/usr/bin/env python3
"""Compute the primary protocol cells FROM predictions, not from scalars.

Loads dataset.jsonl, builds the within-family interior-size split (hold out
the largest 20% of sizes per family by sorted size), trains roofline (ridge on
physical descriptors), GBT (HistGradientBoostingRegressor), and transformer
(small encoder, numeric features) with fixed seeds, and writes
predictions_primary.jsonl with one row per (model, test row):
{model, family, size, block, y_true, y_pred}.

Split rule: for each family, sort rows by (size, block); hold out the last
20% as test. Pool across families. n_train=537, n_test=136.
"""
import json, math, pathlib, sys, random
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import torch, torch.nn as nn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from epic_u2o.model import create_model

HERE = pathlib.Path(__file__).resolve().parent
DATASET = HERE / "dataset.jsonl"
OUT = HERE / "predictions_primary.jsonl"

def phys(r):
    n=r["size"]; b=r["block"]
    bm=2*n*4.0
    fl=n*(64.0 if r["family"]=="matmul" else 1.0)
    ai=fl/bm
    grid=min((n+b-1)//b,65535)
    waves=grid/170.0
    return [math.log1p(n), b, bm, fl, ai, grid, waves]

def main():
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    rows=[json.loads(l) for l in DATASET.read_text().splitlines()]
    # split
    train=[]; test=[]
    for fam in sorted(set(r["family"] for r in rows)):
        fr=sorted([r for r in rows if r["family"]==fam], key=lambda r:(r["size"],r["block"]))
        cut=int(0.8*len(fr))
        train.extend(fr[:cut]); test.extend(fr[cut:])
    assert len(train)==537 and len(test)==136, (len(train),len(test))
    Xtr=np.array([phys(r) for r in train],dtype=np.float32)
    ytr=np.array([r["y"] for r in train],dtype=np.float32)
    Xte=np.array([phys(r) for r in test],dtype=np.float32)
    yte=np.array([r["y"] for r in test],dtype=np.float32)
    mu=Xtr.mean(0); sd=Xtr.std(0); sd[sd==0]=1
    Xtrn=(Xtr-mu)/sd; Xten=(Xte-mu)/sd

    preds=[]
    # roofline (ridge)
    rr=Ridge(); rr.fit(Xtrn,ytr); rp=rr.predict(Xten)
    for r,p in zip(test,rp):
        preds.append({"model":"roofline_94","family":r["family"],"size":r["size"],"block":r["block"],"y_true":float(r["y"]),"y_pred":float(p)})
    # gbt
    g=HistGradientBoostingRegressor(max_iter=300,max_depth=6,learning_rate=0.05,min_samples_leaf=5,random_state=0)
    g.fit(Xtr,ytr); gp=g.predict(Xte)
    for r,p in zip(test,gp):
        preds.append({"model":"gbt_95","family":r["family"],"size":r["size"],"block":r["block"],"y_true":float(r["y"]),"y_pred":float(p)})
    # transformer
    ids_tr=np.zeros((len(train),8),dtype=np.int64); mask_tr=np.ones((len(train),8),dtype=np.int64)
    ids_te=np.zeros((len(test),8),dtype=np.int64); mask_te=np.ones((len(test),8),dtype=np.int64)
    model=create_model(vocab_size=256,max_seq_len=8,d_model=64,n_heads=4,n_layers=2,d_ff=128,numeric_dim=7,use_numeric=True,dual_head=False,fusion_type="concat")
    model.to("cpu")
    opt=torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(5):
        model.train()
        for i in range(0,len(ytr),32):
            sl=slice(i,min(i+32,len(ytr)))
            opt.zero_grad()
            out=model(torch.tensor(ids_tr[sl]), torch.tensor(mask_tr[sl]), torch.tensor(Xtrn[sl]))
            loss=nn.MSELoss()(out, torch.tensor(ytr[sl]).unsqueeze(1))
            loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        out=model(torch.tensor(ids_te),torch.tensor(mask_te),torch.tensor(Xten))
        tp=out.squeeze(1).numpy()
    for r,p in zip(test,tp):
        preds.append({"model":"transformer_91","family":r["family"],"size":r["size"],"block":r["block"],"y_true":float(r["y"]),"y_pred":float(p)})
    # check finite
    assert all(math.isfinite(p["y_pred"]) for p in preds), "non-finite prediction"
    OUT.write_text("\n".join(json.dumps(p) for p in preds)+"\n")
    print(f"wrote {len(preds)} predictions to {OUT}")

if __name__=="__main__":
    main()
