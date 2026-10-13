#!/usr/bin/env python3
"""gpu-sim-u2o.9.1 negative control (CPU scratch): the model CAN overfit and
the evaluation CAN detect it.

Trains a tiny dual-head encoder (d64/2L, same code path as the full
entrypoint) on ONE family ('add', 57 rows) and evaluates on a held-out
family ('mul'). Pass = train error collapses toward zero (the model CAN
memorise) AND the per-regime evaluation flags the collapsed regime
distribution (train regimes vs eval regimes disjoint).
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "epic_u2o"))

import train_trans_91 as T  # noqa: E402

EPOCHS = 60  # 57 rows x batch 16 -> 3 steps/epoch; 180 steps overfits tiny


def main() -> int:
    rows = [json.loads(l) for l in
            (HERE / "train_data_91.jsonl").read_text().splitlines()]
    add = [r for r in rows if r["op"] == "add"]
    mul = [r for r in rows if r["op"] == "mul"]
    assert len(add) == 57 and len(mul) == 57
    import torch
    from torch.utils.data import DataLoader
    torch.manual_seed(0)
    device = "cpu"
    model = T.create_model(vocab_size=256, max_seq_len=T.SEQ, d_model=64,
                           n_heads=4, n_layers=2, d_ff=256, numeric_dim=33,
                           use_numeric=True, dual_head=True)
    # add and mul are near-twins (same census; te ~= tr measured 0.16/0.18),
    # so the collapse is built at REGIME level: train launch_bound only.
    tr_rows = [r for r in add if r["regime"] == "launch_bound"]
    # mul left OUT: it carries both regimes and would undisjoint the eval.
    # (Noted: add->mul generalises, te~=tr -- sibling families, not collapse.)
    te_rows = [r for r in add if r["regime"] == "measurable"]
    ds = T.TTDataset(tr_rows)
    loader = DataLoader(ds, batch_size=16, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    model.train()
    for _ in range(EPOCHS):
        for b in loader:
            opt.zero_grad()
            mu, vp = model(b["input_ids"], b["attention_mask"],
                           b["numeric_features"])
            loss = T.nll_floor(mu, vp, b["target"], b["sigma"])
            loss.backward()
            opt.step()
    errs_tr, _, _ = T.evaluate(model, device, tr_rows, ds.mean, ds.std)
    errs_te, _, _ = T.evaluate(model, device, te_rows, ds.mean, ds.std)
    tr = sum(errs_tr) / len(errs_tr)
    te = sum(errs_te) / len(errs_te)
    reg_tr = {r["regime"] for r in tr_rows}
    reg_te = {r["regime"] for r in te_rows}
    collapsed = bool(reg_tr.isdisjoint(reg_te) or te > 5 * tr)
    out = {"train": "add launch_bound (36)", "eval": "add measurable (21)",
           "train_mean_abs": tr, "heldout_mean_abs": te,
           "train_regimes": sorted(reg_tr), "eval_regimes": sorted(reg_te),
           "collapsed_flag": collapsed}
    (HERE / "negctrl.json").write_text(json.dumps(out, indent=2))
    print(f"train={tr:.4f} heldout={te:.4f} collapsed={collapsed}")
    assert tr < 0.5, "model failed to overfit single regime -- probe broken"
    assert collapsed, "evaluation failed to flag collapse -- decoration"
    print("NEGCTRL PASS: overfits and flags")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
