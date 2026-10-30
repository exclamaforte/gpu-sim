# gpu-sim-elq2: how many 100x8 config rankings survive the unmeasured fabric term?
# Sweeps term_scale.cp over the ±23% band from gpu-sim-b1r6 and reports winner flips.
# Run: python3 experiments/gpu-sim-elq2/fabric_band_sweep.py
import sys
sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
import importlib.util
spec_ = importlib.util.spec_from_file_location("sa", "scripts/scaleout_answers.py")
sa = importlib.util.module_from_spec(spec_); spec_.loader.exec_module(sa)

S = [1/1.233, 1/1.10, 1.0, 1.10, 1.233]   # fabric-term error factors
spec = sa.build_spec(100); n = 800
rows = []
for mname, model in sa.MODELS.items():
    for batch in (1, 8):
        for cname, cp in sa.CONVENTIONS.items():
            best = {}
            for s in S:
                a = sa.answer(spec, model, n, cp*s, batch)
                best[s] = None if not a else (a.get("tp"), a.get("pp"), a.get("cp"),
                                              a.get("dp"), a.get("zero"))
            base = best[1.0]
            flips = [s for s in S if best[s] != base]
            rows.append((mname, cname, batch, base, flips))

print(f"{'model':12} {'convention':20} {'b':>2} {'tp pp cp dp z':>16}  flips_at")
for m, c, b, base, flips in rows:
    bs = "-" if base is None else " ".join(str(x) for x in base)
    print(f"{m:12} {c:20} {b:>2} {bs:>16}  {[round(f,3) for f in flips]}")
print("\nrow flips at any tested factor:", sum(1 for r in rows if r[4]), "/", len(rows))
