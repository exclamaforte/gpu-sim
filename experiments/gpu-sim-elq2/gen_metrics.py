#!/usr/bin/env python3
"""Generate experiments/gpu-sim-elq2/metrics.json.

100x8 winner sensitivity to the unmeasured fabric term (term_scale_cp): sweep
term_scale.cp over [1/1.233, 1/1.10, 1.0, 1.10, 1.233] (1.233 = b1r6
N64_seq32768 flip ratio) and count how many of the 12 published rows change
winner.
"""
import json, sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import scaleout_answers as sa

FACTORS = [1 / 1.233, 1 / 1.10, 1.0, 1.10, 1.233]
spec = sa.build_spec(100)
n = 800

rows = []
flipped = 0
for mname, model in sa.MODELS.items():
    for batch in (1, 8):
        for cname, scale in sa.CONVENTIONS.items():
            winners = {}
            for f in FACTORS:
                r = sa.answer(spec, model, n, scale * f, batch)
                winners[round(f, 6)] = r["config"] if r else None
            base = winners[1.0]
            flips = [f for f in FACTORS if winners[round(f, 6)] != base and f != 1.0]
            if flips:
                flipped += 1
            rows.append({
                "model": mname,
                "convention": cname,
                "batch": batch,
                "base_winner": base,
                "base_margin_pct": sa.answer(spec, model, n, scale, batch)["margin_pct"],
                "flips_at": [round(f, 6) for f in flips],
                "flipped": bool(flips),
            })

metrics = {
    "bead": "gpu-sim-elq2",
    "gate": "n/a (CPU-only fabric-term sweep; verified by gate --fast on the landing tree)",
    "fabric": "100x8",
    "n_gpus": n,
    "factors": FACTORS,
    "n_rows": len(rows),
    "n_rows_flipping": flipped,
    "rows": rows,
}
out = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else (pathlib.Path(__file__).resolve().parent / "metrics.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(metrics, indent=2) + "\n")
print(json.dumps({"n_rows": len(rows), "n_flipping": flipped}, indent=2))
